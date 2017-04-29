from . import poly
import numpy as np 
import os
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d

__version__ = 1.1

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
    def __init__(self,GRISM_CONF,DIRFILTER=None,cross_filter=None):
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

            wmin = np.min(self.SENS_data[order][0][self.SENS_data[order][1]!=0])
            wmax = np.max(self.SENS_data[order][0][self.SENS_data[order][1]!=0])
            self.WRANGE[order] = [wmin,wmax]
 
            if cross_filter!=None:
                # get name of filter bandpass from config file
                filter_filename = self._get_value("FILTER_%s" % (cross_filter))
                filter_filename = os.path.join(self.GRISM_CONF_PATH,filter_filename)
                passband_tab = Table.read(filter_filename,format="ascii.no_header",data_start=1)
                # Convert bandpass to angstrom
                passband_tab['col1'] = passband_tab['col1']*10000
                self._apply_passband(order,passband_tab,threshold=1e-4)

#            if not pool:
            self.SENS[order] = interp1d_picklable(self.SENS_data[order][0],self.SENS_data[order][1],bounds_error=False,fill_value=0.)

            
            self.XRANGE[order] = self._get_value("XRANGE_%s" % (order),type=float)
            self.YRANGE[order] = self._get_value("YRANGE_%s" % (order),type=float)

    def _apply_passband(self,order,passband_tab,threshold):
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
            print("Sensitivity and filter passband do not ovelap. Check units...")
        
        self.SENS_data[order][1] = np.asarray(ys)
        self.SENS_data[order][0] = np.asarray(xs)

        wmin = np.min(self.SENS_data[order][0][self.SENS_data[order][1]>np.max(self.SENS_data[order][1])*threshold])
        wmax = np.max(self.SENS_data[order][0][self.SENS_data[order][1]>np.max(self.SENS_data[order][1])*threshold])
        #print "Bandpass reduced to ===>",wmin,wmax
        self.WRANGE[order] = [wmin,wmax]

        self.SENS[order] = interp1d_picklable(self.SENS_data[order][0],self.SENS_data[order][1],bounds_error=False,fill_value=0.)

        return

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

    #def DLDX(self,order,x0,y0,t):
    #    return self.DDISPL(order,x0,y0,t)/self.DDISPX(order,x0,y0,t)

    #def DLDY(self,order,x0,y0,t):
    #    return self.DDISPL(order,x0,y0,t)/self.DDISPY(order,x0,y0,t)

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

def testing():
    
    # NIRISS
    gr, filter = 'C', 'F150W'
    order = 'A'
    conf = grismconf.grismconf.Config('GR150{0}.{1}.t.conf'.format(gr, filter))
    
    t = np.linspace(0,1,10)
    x0, y0 = 1536, 1536

    # dx, forward polynomial
    dx = conf.DISPX(order, x0, y0, t)
    dxp = forward(conf._DISPX_data[order], x0, y0, t)
    np.testing.assert_array_almost_equal(dx, dxp, decimal=4)
    
    # inverse
    ti = conf.INVDISPX(order, x0, y0, dx)
    tip = inverse(conf._DISPX_data[order], x0, y0, dx)
    
    np.testing.assert_array_almost_equal(t, ti, decimal=4)
    np.testing.assert_array_almost_equal(ti, tip, decimal=4)
    
    # Derivative is different, correct for `deriv`
    ddx = conf.DDISPX(order, x0, y0, t) 
    ddxp = deriv(conf._DISPX_data[order], x0, y0, t) 
    
    print('DDISPX:',ddx)  # single float
    print('deriv:', ddxp) # evaluated at all t 
    
    
def forward(coeffs, x0, y0, t):
    """Forward evaluation of field-dependent polynomials f(x0,y0,t)
    
    Parameters
    ----------
    x0, y0 : float 
        Coordinate to evaluate the field dependent coefficients
    
    t : float or array-like
        Independent variable at which to evaluate the polynomial
    
    Returns
    -------
    d : float or array-like
        Evaluated value at f(x0,y0,t).
    
    """
    a_i = field_dependent(x0, y0, coeffs)
    d = np.polyval(a_i[::-1], t)
    return d

def deriv(coeffs, x0, y0, t):
    """df/dt of the field-dependent polynomials f(x0,y0,t)
    
    Parameters
    ----------
    x0, y0 : float 
        Coordinate to evaluate the field dependent coefficients
    
    t : float or array-like
        Independent variable at which to evaluate the polynomial
    
    Returns
    -------
    d : float or array-like
        Evaluated derivative at df(x0,y0,t)/dt.
    
    """
    a_i = field_dependent(x0, y0, coeffs)
    if len(a_i) == 1:
        return t*0.
    
    power = np.arange(len(a_i))
    a_i_prime = (a_i*power)[1::][::-1]
    d = np.polyval(a_i_prime, t)
    return d
    
def inverse(coeffs, x0, y0, d):
    """Inverse of field-dependent polynomials, d = f(x0,y0,t)
    
    Parameters
    ----------
    x0, y0 : float 
        Coordinate to evaluate the field dependent coefficients
    
    d : float or array-like
        Dependent variable at which to compute the roots, `t`, i.e., 
        `d = f(x0,y0,t)`.
    
    Returns
    -------
    t : float or array-like
        Independent variable.
        
    .. note::
        
        Current implementation is analytic for polynomial orders 0 and 1.  
        Uses `~numpy.poly1d` to compute roots of higher order polynomials and
        currently returns all roots, i.e., the two roots for a quadratic
        function.
    
    """
    from scipy import polyval
    a_i = field_dependent(x0, y0, coeffs)
    if len(a_i) == 1:
        t = d*0.+a_i
    elif len(a_i) == 2:
        t = (d-a_i[0])/a_i[1]
    else:
        # Check for higher order functions
        p = np.poly1d(a_i[::-1])
        t = np.squeeze([(p-di).roots[0] for di in d])

    return t
    
def field_dependent(x0, y0, coeffs):
    """aXe-like field-dependent coefficients
    
    See the `aXe manual <http://axe.stsci.edu/axe/manual/html/node7.html#SECTION00721200000000000000>`_ for a description of how the field-dependent coefficients are specified.
    
    Parameters
    ----------
    x0, y0 : float or array-like
        Coordinate to evaluate the field dependent coefficients, where
    
    coeffs : array-like
        Field-dependency coefficients
    
    Returns
    -------
    a : float or array-like
        Evaluated field-dependent coefficients
    
    .. note::
    
    Taken from `~grizli`: 
    https://github.com/gbrammer/grizli/blob/master/grizli/grismconf.py#L127.
    
    """
    ## number of coefficients for a given polynomial order
    ## 1:1, 2:3, 3:6, 4:10, order:order*(order+1)/2
    if isinstance(coeffs, float):
        order = 1
    else:
        order = int(-1+np.sqrt(1+8*coeffs.shape[-1])) // 2

    ## Build polynomial terms array
    ## $a = a_0+a_1x_i+a_2y_i+a_3x_i^2+a_4x_iy_i+a_5yi^2+$ ...
    xy = []
    for p in range(order):
        for px in range(p+1):
            #print 'x**%d y**%d' %(p-px, px)
            xy.append(x0**(p-px)*y0**(px))

    ## Evaluate the polynomial, allowing for N-dimensional inputs
    a = np.sum((np.array(xy).T*coeffs).T, axis=0)

    return a
