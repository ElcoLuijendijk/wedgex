import numpy as np
import astropy.units as u
import shapely
from shapely.geometry import LineString, Point



def calculate_closure_temp(time, temp, thermochron_system, min_gradient=1e-7*u.K/u.year):
    
    """
    calculate closure temperatures using Dodson's (1973) equations
    
    shamelessly copied from glide's fortran code:
    https://github.com/cirederf13/glide/blob/master/transient_geotherm/src/closure_temps.f90
    https://github.com/cirederf13/glide/blob/master/transient_geotherm/src/dodson.f90
    
    parameters based on Reiners and Brandon (2006, https://doi.org/10.1146/annurev.earth.34.031405.125202)
    
    """
    
    #Myr = 3600.*24.*365.25e6
    
    # these values are taken from reiners' review paper and correspond to Ea (x1000.) and omega (x secinyr)      
    if thermochron_system == 'AFT':
        # here are Ea and D0/a2 values for AFT from ketcham. 1999
        # taken from reiners 2004
        energy = 147 * 1e3 * u.J / u.mol
        geom = 1
        diff = 2.05e6 / u.s
    elif thermochron_system == 'ZFT':
        #here are Ea and D0/a2 values for ZFT from reiners2004
        #taken from reiners 2004
        energy = 208 * 1e3 * u.J / u.mol #!/(4.184*1.e3)
        geom = 1
        diff = 4.0e8 / u.s
        #energy=224.d3
        #geom=1.d0
        #diff=1.24e8*3600.*24.*365.25e6
        #diff = 1.24e8*3600.*24.*365.25e6
    elif thermochron_system == 'AHe':
        # here are Ea and D0/a2 values for AHe from Farley et al. 2000
        # taken from reiners 2004
        energy = 138 * 1e3 * u.J / u.mol
        geom = 1
        diff = 7.64e7 / u.s
    elif thermochron_system == 'ZHe':
        #here are Ea and D0/a2 values for ZHe from reiners2004
        #taken from reiners 2004
        energy = 169 * 1e3 * u.J / u.mol #!/(4.184*1.e3)
        geom = 1
        diff = 7.03e5 / u.s
        #energy=178.d3
        #geom=1.d0
        #diff=7.03d5*3600.d0*24.d0*365.25d6

    #the following are for argon argon, might be a bit much for most, 51,52,53 in glide
    elif thermochron_system == 'hbl':
        #here are Ea and D0/a2 values for hbl from harrison81
        #taken from reiners 2004
        energy=268 * 1e3 * u.J / u.mol
        geom=1
        diff=1320 / u.s
    elif thermochron_system == 'MAr':
        # here are Ea and D0/a2 values for mus from hames&bowring1994,robbins72
        # taken from reiners 2004
        energy = 180 * 1e3 * u.J / u.mol
        geom=1
        diff=3.91 / u.s
        
        # note correction in manuscript Reiners (2006):
        #diff = 17.2 / u.s
        
        # Alternative values in Harrison et al. (2009, https://doi.org/10.1016/j.gca.2008.09.038)
        # as cited in McDonald et al.  (2018, https://doi.org/10.1111/ter.12390)
        #energy = 263592 * u.J / u.mol
        #geom = 1
        #D0 = 2.3e-4 * u.m**2 / u.s
        #a = 100.0 * 1e-6 * u.m
        #diff = D0 / a**2
        
    elif thermochron_system == 'bio':
        #here are Ea and D0/a2 values for bio from grove&harrison1996
        #taken from reiners 2004
        energy = 197 * 1e3 * u.J / u.mol
        geom = 1
        diff = 733. / u.s

    #energy = 147.0 * 1e3 * u.J / u.mol
    #geom = 1.0
    #diff = 2.05e6 / u.s

    r = 8.314 * u.J / (u.K * u.mol)

    cooling = np.gradient(temp, time)
    
    #print(cooling)
    
    ind = cooling < min_gradient
    cooling[ind] = min_gradient
    
    #print(cooling.mean())
    
    tau = r * temp**2 / (energy * cooling)
    Tc = energy / (r * np.log(geom * tau * diff))
    
    return Tc


def calculate_closure_age(time, temp, thermochron_system, verbose=False):
    
    """
    first calculate closure temperatures and then find the intersection between the closure temperature vs time curve 
    and the temperature vs time curve
    """
    
    Tc = calculate_closure_temp(time, temp, thermochron_system)
    
    if verbose is True:
        print('closure temp = ', Tc)
    
    if Tc.max() < temp.min() or Tc.min() > temp.max():
        #print('warning, cooling temp outside of range of thermochron temps')
        #print('range: ', Tc.min(), Tc.max())
        #raise ValueError
        
        return np.nan, np.nan
    
    else:
        xy1 = np.vstack([time.to(u.year).value, temp.value]).T
        xy2 =  np.vstack([time.to(u.year).value, Tc.value]).T
        line1 = LineString(xy1)
        line2 = LineString(xy2)

        int_pt = line1.intersection(line2)
        if int_pt.type == 'MultiPoint':
            xi, yi = int_pt[0].x, int_pt[0].y
        elif int_pt.type == 'LineString':
            if len(int_pt.coords) > 0:
            
                xi, yi = int_pt.coords.xy[:, 0], int_pt.coords.xy[:, 1]
            else:
                return np.nan, np.nan
        else:
            xi, yi = int_pt.x, int_pt.y

        age = xi * u.year
        Tc_int = yi * u.K

        return age, Tc_int