# Note: Code provided by R. Ryan.

from jwst import datamodels
import asdf
import os, sys
import requests
import numpy as np 
from astropy.modeling import polynomial
from astropy.io import fits
from astropy.table import Table

def fetch_reffile(filename, overwrite=True,show=True):
    crdsurl = f"https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{filename}"

    download = False
    if os.path.exists(filename):
        if overwrite:
            # do it
            download = True
    else:
        # do it
        download = True


    if (os.path.exists(filename) and overwrite) or not os.path.exists(filename):
        if show:
            print(f"Fetching the file {filename}")
        r = requests.get(crdsurl, stream=True) # params=params, headers=headers, stream=True)
        r.raise_for_status()
        with open(filename, "wb") as fobj:
            for chunk in r.iter_content(chunk_size=1_024_000):
                fobj.write(chunk)
    else:
        if show:
            print(f'Using local copy of {filename}')


def reformat_poly(obj):
    '''
    Function to transform an astropy.Polynomial object into an list() that matches a given row of a grismconf file
    '''
    
    coefs = list(np.zeros(len(obj.parameters), dtype=float))
    n = len(coefs)

    if isinstance(obj, polynomial.Polynomial1D):
        for i in range(n):
            coefs[i] = [getattr(obj, f'c{i}').value]
    elif isinstance(obj, polynomial.Polynomial2D):
        m = int(np.sqrt(8*n+1)-1)//2
        i = 0
        for j in range(m):
            for k in range(j+1):
                coefs[i] = getattr(obj, f'c{j-k}_{k}').value
                i += 1
    else:
        raise NotImplementedError(type(obj))
        
    return coefs

def get_sensitivity(wfss_file,order=1,show=False):
    """Fetch and process the sensitivity file for this observation. This function cleans up the content of 
    the calibration file and changes the units of the sensitivity to be in flam per DN/s"""
    from scipy.interpolate import interp1d
    
    # We need to get the pixel size of the detector. We also get the PUPIL and FILTER name
    with fits.open(wfss_file) as fin:
        pixel_area = fin[1].header['PIXAR_SR']
        pupil = fin[0].header['PUPIL']
        filter = fin[0].header['FILTER']

    m = datamodels.open(wfss_file)

    sensitivity_file = m.meta.ref_file.photom.name[7:]
    fetch_reffile(sensitivity_file,overwrite=False,show=False)

    tab = Table.read(sensitivity_file)
    ok = (tab['filter']==filter) & (tab['pupil']==pupil) & (tab['order']==order)
    w,s, = np.asarray(tab[ok][0]['wavelength']),np.asarray(tab[ok][0]['relresponse'])
    photmjsr = tab[ok][0]['photmjsr']
    ok = np.nonzero(w)
    w = w[ok]
    s = s[ok]

    # The sensitivity is by default in units of Mjy per SR per DN/s (per pixel) which we convert to
    # the more traditional value of erg/s/cm^2/A per DN/s
    c = 29_979_245_800.0 
    s2 = (w*1e4)/c * (w/1e8) / (s*photmjsr*1e6*1e-23*pixel_area) * 10000

    if show:
        plt.plot(w,s2)
        plt.xlabel(r"Wavelength ($\mu m$)")
        plt.ylabel(r"DN/s per erg/s/cm^2/$\AA$")
        plt.grid()

    return w,s2

def specwcs_poly(wfss_file,order=1):
    with datamodels.open(wfss_file) as dm:
        try:
            specwcs = dm.meta.ref_file.specwcs.name[7:]
            photom = dm.meta.ref_file.photom.name[7:]
            pixarea = dm.meta.photometry.pixelarea_steradians
        except:
            raise NameError('Failed to find WFSS WCS information in input file. Make sure that the JWST pipeline wcs_assign and photom steps were applied.')

    _DISPX_data = {}
    _DISPY_data = {}
    _DISPL_data = {}
    SENS_data = {}
    with datamodels.open(wfss_file) as tree:
        t = tree['meta'].wcs .get_transform(from_frame='detector', to_frame='grism_detector')[-1]
        for g,order in enumerate(t.orders):
            sorder = '{0:+}'.format(order)
            _DISPX_data[sorder] = np.array([reformat_poly(p2d) for p2d in t.xmodels[g]])
            if len(t.xmodels[g])==1: _DISPX_data[sorder] = _DISPX_data[sorder][0]

            _DISPY_data[sorder] = np.array([reformat_poly(p2d) for p2d in t.ymodels[g]])
            if len(t.ymodels[g])==1: _DISPY_data[sorder] = _DISPY_data[sorder][0]

            _DISPL_data[sorder] = np.array([reformat_poly(p2d) for p2d in t.lmodels[g]])
            if len(t.lmodels[g])==1: _DISPL_data[sorder] = _DISPL_data[sorder][0]

            SENS_data[sorder] = get_sensitivity(wfss_file,order=order)


    return _DISPX_data,_DISPY_data,_DISPL_data,SENS_data