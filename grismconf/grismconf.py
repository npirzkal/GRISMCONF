from . import poly
import numpy as np 
import os
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d

__version__ = 1.2

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

    def _getstate__(self):
        return self.xi, self.yi, self.args

    def __setstate__(self, state):
        self.f = interp1d(state[0], state[1], **state[2])

class Config(object):
    """Class to read and hold GRISM configuration info"""
    def __init__(self,GRISM_CONF,DIRFILTER=None):
        """Read in Grism Configuration file and populate various things"""
        self.__version__=__version__
        self.GRISM_CONF = open(GRISM_CONF).readlines()
        self.GRISM_CONF_PATH = os.path.dirname(GRISM_CONF)
        self.GRISM_CONF_FILE = os.path.basename(GRISM_CONF)

        self.orders = self._get_orders()
        self._DISPX_data = {}
        self._DISPY_data = {}
        self._DISPL_data = {}

        self._DISPX_polyname = {}
        self._DISPY_polyname = {}
        self._DISPL_polyname = {}

        self.SENS = {}
        self.SENS_data = {}

        # Wavelength range of the grism
        self.WRANGE = {}

        # Extent of FOV in detector pixel
        self.XRANGE = {}
        self.YRANGE = {}

        if DIRFILTER!=None:
            # We get the wedge offset values for this direct filter
            r = self._get_value("WEDGE_%s" % (DIRFILTER),type=float)
            self.wx = r[0]
            self.wy = r[1]
        else:
            self.wx = 0.
            self.wy = 0.

        for order in self.orders:    
            self._DISPX_data[order] = self._get_parameters("DISPX",order)
            self._DISPY_data[order] = self._get_parameters("DISPY",order)
            self._DISPL_data[order] = self._get_parameters("DISPL",order)
            self.SENS[order] = self._get_sensitivity(order)
            
            self._DISPX_polyname[order] = np.shape(self._DISPX_data[order])
            self._DISPY_polyname[order] = np.shape(self._DISPY_data[order])
            self._DISPL_polyname[order] = np.shape(self._DISPL_data[order])

            self.SENS_data[order] = self._get_sensitivity(order)

            vg = self.SENS_data[order][1]>np.max(self.SENS_data[order][1])*1e-3
            wmin = np.min(self.SENS_data[order][0][vg])
            wmax = np.max(self.SENS_data[order][0][vg])
            self.WRANGE[order] = [wmin,wmax]

            self.SENS[order] = interp1d_picklable(self.SENS_data[order][0],self.SENS_data[order][1],bounds_error=False,fill_value=0.)
        
            self.XRANGE[order] = self._get_value("XRANGE_%s" % (order),type=float)
            self.YRANGE[order] = self._get_value("YRANGE_%s" % (order),type=float)

    @staticmethod
    def _rotate_coords(dx, dy, theta=0, origin=[0,0]):
        """Rotate cartesian coordinates CW about an origin
        
        Parameters
        ----------
        dx, dy : float or `~numpy.ndarray`
            x and y coordinages
                    
        theta : float
            CW rotation angle, in radians
            
        origin : [float,float]
            Origin about which to rotate
        
        Returns
        -------
        dxr, dyr : float or `~numpy.ndarray`
            Rotated versions of `dx` and `dy`
            
        """
        _mat = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

        rot = np.dot(np.array([dx-origin[0], dy-origin[1]]).T, _mat)
        dxr = rot[:,0]+origin[0]
        dyr = rot[:,1]+origin[1]
        return dxr, dyr

    def DISPL(self,order,x0,y0,t):
        """DISPL() returns the wavelength l = DISPL(x0,y0,t) where x0,y0 is the 
        position on the detector and 0<t<1"""
        return poly.POLY[self._DISPL_polyname[order]](self._DISPL_data[order],x0,y0,t)

    def DDISPL(self,order,x0,y0,t):
        """DDISPL returns the wavelength 1st derivative with respect to t  l =  d(DISPL(x0,y0,t))/dt where x0,y0 is the position on the detector and 0<t<1"""
        return poly.DPOLY[self._DISPL_polyname[order]](self._DISPL_data[order],x0,y0,t)

    def DISPXY(self, order, x0, y0, t, theta=0):
        """Return both `x` and `y` coordinates of a rotated trace
        
        Parameters
        ----------
        order : str
            Order string
            
        x0, y0 : float
            Reference position (i.e., in direct image)

        t : float or `~numpy.ndarray`
            Parameter where to evaluate the trace
            
        theta : float
            CW rotation angle, in radians
        
        Returns
        -------
        dxr, dyr : float or `~np.ndarray`
            Rotated trace coordinates as a function of `t`

        """
        dx = -self.wx + poly.POLY[self._DISPX_polyname[order]](self._DISPX_data[order],x0,y0,t)
        dy = -self.wy + poly.POLY[self._DISPY_polyname[order]](self._DISPY_data[order],x0,y0,t)

        if theta != 0:
            dxr, dyr = self._rotate_coords(dx, dy, theta=theta, origin=[0,0])
            return dxr, dyr
        else:
            return dx, dy
            
    
    def INVDISPXY(self, order, x0, y0, dx=None, dy=None, theta=0, t0=np.linspace(0,1,10)):
        """Return independent variable `t` along rotated trace
        
        Parameters
        ----------
        order : str
            Order string
            
        x0, y0 : float
            Reference position (i.e., in direct image)

        dx : float, `~numpy.ndarray` or None
            `x` coordinate in *rotated* trace where to evaluate the trace
            independent variable `t`.
            
        dy : float, `~numpy.ndarray` or None
            Same as `dx` but evaluate along 'y' axis.
        
        t0 : `~np.ndarray`
            Independent variable location where to evaluate the rotated trace.
            For low-order trace shapes, this can be coarsely sampled as 
            in the default.
            
        Returns
        -------
        tr : float or `~np.ndarray`
            Independent variable `t` evaluated on the rotated trace at
            `dx` or `dy`.

        .. note::
        
        Order of execution is first check if `dx` supplied.  If not, then
        check `dy`.  And if both are None, then return None (do nothing).
        
        """
        if dx is not None:
            xr, yr = self.DISPXY(order, x0, y0, t0, theta=theta)
            so = np.argsort(xr)
            tr = np.interp(dx, xr[so], t0[so])
            return tr
        
        if dy is not None:
            xr, yr = self.DISPXY(order, x0, y0, t0, theta=theta)
            so = np.argsort(yr)
            tr = np.interp(dy, yr[so], t0[so])
            return tr
                 
        return None
        
    def DISPX(self,order,x0,y0,t,theta=0):
        """DISPX() eturns the x offset x'-x = DISPL(x0,y0,t) where x0,y0 is the 
        position on the detector, x'-x is the difference between direct and grism image x-coordinates and 0<t<1"""
        dx = -self.wx + poly.POLY[self._DISPX_polyname[order]](self._DISPX_data[order],x0,y0,t)
            
        return  dx

    def DDISPX(self,order,x0,y0,t):
        """DDISPX returns the  1st derivative of DISPX() with respect to t  d(x'-x)/dt =  d(DISPX(x0,y0,t))/dt where x0,y0 is the position on the detector and 0<t<1"""
        return  poly.DPOLY[self._DISPX_polyname[order]](self._DISPX_data[order],x0,y0,t)

    def DISPY(self,order,x0,y0,t):
        """DISPY() eturns the x offset 'y-y = DISPL(x0,y0,t) where x0,y0 is the 
        position on the detector, y'-y is the difference between direct and grism image y-coordinates and 0<t<1"""
        return  -self.wy + poly.POLY[self._DISPY_polyname[order]](self._DISPY_data[order],x0,y0,t)

    def DDISPY(self,order,x0,y0,t):
        """DDISPY returns the  1st derivative of DISPY() with respect to t  d(y'-y)/dt =  d(DISPY(x0,y0,t))/dt where x0,y0 is the position on the detector and 0<t<1"""
        return poly.DPOLY[self._DISPY_polyname[order]](self._DISPY_data[order],x0,y0,t)

    def INVDISPL(self,order,x0,y0,l):
        """INVDISL() returns the t values corresponding to a given wavelength l, t = INVDISPL(x0,y0,l)"""
        return poly.INVPOLY[self._DISPL_polyname[order]](self._DISPL_data[order],x0,y0,l)

    def INVDISPX(self,order,x0,y0,dx):
        """INVDISPX returns the x value corresponding to a given wavelength l, t = INVDISPL(x0,y0,l)"""
        return poly.INVPOLY[self._DISPX_polyname[order]](self._DISPX_data[order],x0,y0,dx+self.wx)

    def INVDISPY(self,order,x0,y0,dy):
        """INVDISPY returns the y value corresponding to a given wavelength l, t = INVDISPL(x0,y0,l)"""
        return poly.INVPOLY[self._DISPY_polyname[order]](self._DISPY_data[order],x0,y0,dy+self.wy)    

    def _get_orders(self):
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

    def _get_parameters(self,name,order,str_fmt="%s_%s_"):
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
                    print("Wrong format for ",GRISM_CONF,name,order)
                    sys.exit(10)
                vals = [float(ww) for ww in ws[1:]]
                arr[i,0:m] = vals

        return arr            

    def _get_value(self,str,type=None):
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


    def _get_sensitivity(self,order):
        """Helper function that looks for the name of the sensitivity file, reads it and
        stores the content in a simple list [WAVELENGTH, SENSITIVITY]."""
        fname = os.path.join(self.GRISM_CONF_PATH,self._get_value("SENSITIVITY_%s" % (order)))
        fin = fits.open(fname)
        wavs = fin[1].data.field("WAVELENGTH")[:]
        sens = fin[1].data.field("SENSITIVITY")[:]
        fin.close()        
        # Fix for cases where sensitivity is not zero on edges
        sens[0:2] = 0.
        sens[-2:] = 0.
                
        return [wavs,sens]    
    

