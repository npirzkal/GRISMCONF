import poly
import numpy as np 
import os
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d


class interp1d_picklable:
    """ class wrapper for piecewise linear function
    """
    def __init__(self, xi, yi, **kwargs):
        self.xi = xi
        self.yi = yi
        self.args = kwargs
        self.f = interp1d(xi, yi, **kwargs)

    def __call__(self, xnew):
        return self.f(xnew)

    def __getstate__(self):
        return self.xi, self.yi, self.args

    def __setstate__(self, state):
        self.f = interp1d(state[0], state[1], **state[2])



class Config(object):
    """Class to read and hold GRISM configuration info"""
    def __init__(self,GRISM_CONF,DIRFILTER=None,passband_tab=None):
        """Read in Grism Configuration file and populate various things"""
        self.GRISM_CONF = open(GRISM_CONF).readlines()
        self.GRISM_CONF_PATH = os.path.split(GRISM_CONF)[0]

        self.orders = self.__get_orders()
        self.DISPX_DATA = {}
        self.DISPY_DATA = {}
        self.DISPL_DATA = {}

        self.DISPX_POLYNAME = {}
        self.DISPY_POLYNAME = {}
        self.DISPL_POLYNAME = {}

        self.SENS = {}
        self.SENS_data = {}

        # Wavelength range of the grism
        self.WRANGE = {}

        # Extent of FOV in detector pixel
        self.XRANGE = {}
        self.YRANGE = {}

        if DIRFILTER!=None:
            # We get the wedge offset values for this direct filter
            r = self.__get_value("WEDGE_%s" % (DIRFILTER),type=float)
            self.wx = r[0]
            self.wy = r[1]
        else:
            self.wx = 0.
            self.wy = 0.

        for order in self.orders:    
            self.DISPX_DATA[order] = self.__get_parameters("DISPX",order)
            self.DISPY_DATA[order] = self.__get_parameters("DISPY",order)
            self.DISPL_DATA[order] = self.__get_parameters("DISPL",order)
            self.SENS[order] = self.__get_sensitivity(order)
            
            self.DISPX_POLYNAME[order] = np.shape(self.DISPX_DATA[order])
            self.DISPY_POLYNAME[order] = np.shape(self.DISPY_DATA[order])
            self.DISPL_POLYNAME[order] = np.shape(self.DISPL_DATA[order])

            self.SENS_data[order] = self.__get_sensitivity(order)
            if passband_tab!=None:
                self.__apply_passband(order,passband_tab)

#            if not pool:
            self.SENS[order] = interp1d_picklable(self.SENS_data[order][0],self.SENS_data[order][1],bounds_error=False,fill_value=0.)

            # To do: Add direct filter trnasmssion here
            self.WRANGE[order] = [np.min(self.SENS_data[order][0]),np.max(self.SENS_data[order][0])]
 
            self.XRANGE[order] = self.__get_value("XRANGE_%s" % (order),type=float)
            self.YRANGE[order] = self.__get_value("YRANGE_%s" % (order),type=float)

    def __apply_passband(self,order,passband_tab):
        """A helper function that applies an additional passband to the existing sensitivity. This modifies self.SENS_data and also recompute the interpolation function stored in self.SENS"""

        # Apply grism sensitibity to filter... i.e. use filter as wavelength basis
        fs = interp1d_picklable(self.SENS_data[order][0],self.SENS_data[order][1],bounds_error=False,fill_value=0.)

        xs = []
        ys = []
        overlap = 0
        for i,l in enumerate(np.array(passband_tab["col1"])):
            xs.append(l)
            ys.append(passband_tab["col2"][i] * fs(l))
            if fs(l)>0:
                overlap = 1
        if overlap==0:
            print "Sensitivity and filter passband do not ovelap. Check units..."
        
        self.SENS_data[order][1] = np.asarray(ys)
        self.SENS_data[order][0] = np.asarray(xs)

        self.SENS[order] = interp1d_picklable(self.SENS_data[order][0],self.SENS_data[order][1],bounds_error=False,fill_value=0.)

        return



    def DISPL(self,order,x0,y0,t):
        """DISPL() returns the wavelength l = DISPL(x0,y0,t) where x0,y0 is the 
        position on the detector and 0<t<1"""
        return poly.POLY[self.DISPL_POLYNAME[order]](self.DISPL_DATA[order],x0,y0,t)

    def DDISPL(self,order,x0,y0,t):
        """DDISPL returns the wavelength 1st derivative with respect to t  l =  d(DISPL(x0,y0,t))/dt where x0,y0 is the position on the detector and 0<t<1"""
        return poly.DPOLY[self.DISPL_POLYNAME[order]](self.DISPL_DATA[order],x0,y0,t)

    def DISPX(self,order,x0,y0,t):
        """DISPX() eturns the x offset x'-x = DISPL(x0,y0,t) where x0,y0 is the 
        position on the detector, x'-x is the difference between direct and grism image x-coordinates and 0<t<1"""
        return  -self.wx + poly.POLY[self.DISPX_POLYNAME[order]](self.DISPX_DATA[order],x0,y0,t)

    def DDISPX(self,order,x0,y0,t):
        """DDISPX returns the  1st derivative of DISPX() with respect to t  d(x'-x)/dt =  d(DISPX(x0,y0,t))/dt where x0,y0 is the position on the detector and 0<t<1"""
        return  poly.DPOLY[self.DISPX_POLYNAME[order]](self.DISPX_DATA[order],x0,y0,t)

    def DISPY(self,order,x0,y0,t):
        """DISPY() eturns the x offset 'y-y = DISPL(x0,y0,t) where x0,y0 is the 
        position on the detector, y'-y is the difference between direct and grism image y-coordinates and 0<t<1"""
        return  -self.wy + poly.POLY[self.DISPY_POLYNAME[order]](self.DISPY_DATA[order],x0,y0,t)

    def DDISPY(self,order,x0,y0,t):
        """DDISPY returns the  1st derivative of DISPY() with respect to t  d(y'-y)/dt =  d(DISPY(x0,y0,t))/dt where x0,y0 is the position on the detector and 0<t<1"""
        return poly.DPOLY[self.DISPY_POLYNAME[order]](self.DISPY_DATA[order],x0,y0,t)

    def INVDISPL(self,order,x0,y0,l):
        """INVDISL() returns the t values corresponding to a given wavelength l, t = INVDISPL(x0,y0,l)"""
        return poly.INVPOLY[self.DISPL_POLYNAME[order]](self.DISPL_DATA[order],x0,y0,l)

    def INVDISPX(self,order,x0,y0,dx):
        """INVDISPX returns the x value corresponding to a given wavelength l, t = INVDISPL(x0,y0,l)"""
        return poly.INVPOLY[self.DISPX_POLYNAME[order]](self.DISPX_DATA[order],x0,y0,dx+self.wx)

    def INVDISPY(self,order,x0,y0,dy):
        """INVDISPY returns the y value corresponding to a given wavelength l, t = INVDISPL(x0,y0,l)"""
        return poly.INVPOLY[self.DISPY_POLYNAME[order]](self.DISPY_DATA[order],x0,y0,dy+self.wy)    

    #def DLDX(self,order,x0,y0,t):
    #    return self.DDISPL(order,x0,y0,t)/self.DDISPX(order,x0,y0,t)

    #def DLDY(self,order,x0,y0,t):
    #    return self.DDISPL(order,x0,y0,t)/self.DDISPY(order,x0,y0,t)

    def __get_orders(self):
        """A helper function that parses the config file and finds all the Orders/BEAMS.
        Simply looks for the BEAM_ keywords"""
        orders = []

        # Get orders 
        for l in self.GRISM_CONF:
            k = "BEAM_"
            if l[0:len(k)]==k:
                ws = l.split()
                order = ws[0].split("_")[-1]
                orders.append(order)
        return orders

    def __get_parameters(self,name,order,str_fmt="%s_%s_"):
        """Return the 2D polynomial array stored in the config file"""
        str = str_fmt % (name,order)
        # Find out how many we have to store
        n = 0
        m = 0
        for l in self.GRISM_CONF:
            if l[0]=="#": continue
            ws = l.split()
            if len(ws)>0 and str in ws[0]:
                i = ws[0].split(str)[-1]
                n = n + 1
                m = len(ws)-1

        arr = np.zeros((n,m))

        for l in self.GRISM_CONF:
            ws = l.split()
            if len(ws)>0 and str in ws[0]:
                i = int(ws[0].split(str)[-1])
                if len(ws)-1 !=m:
                    print "Wrong format for ",GRISM_CONF,name,order
                    sys.exit(10)
                vals = [float(ww) for ww in ws[1:]]
                arr[i,0:m] = vals

        return arr            

    def __get_value(self,str,type=None):
        """Helper function to simply return the value for a simple keyword parameters
        in the config file."""
        
        for l in self.GRISM_CONF:
            ws = l.split()
            if len(ws)>0 and ws[0]==str:
                if len(ws)==2:
                    if type==None:
                        return ws[1]
                    elif type==float:
                        return float(ws[1])
                else:
                    if type==None:
                        return ws[1:]
                    elif type==float:
                        return [float(x) for x in ws[1:]]

        return None


    def __get_sensitivity(self,order):
        """Helper function that looks for the name of the sensitivity file, reads it and
        stores the content in a simple list [WAVELENGTH, SENSITIVITY]."""
        fname = os.path.join(self.GRISM_CONF_PATH,self.__get_value("SENSITIVITY_%s" % (order)))
        fin = fits.open(fname)
        wavs = fin[1].data.field("WAVELENGTH")[:]
        sens = fin[1].data.field("SENSITIVITY")[:]
        fin.close()        
        # Fix for cases where sensitivity is not zero on edges
        sens[0:2] = 0.
        sens[-2:] = 0.
                
        return [wavs,sens]