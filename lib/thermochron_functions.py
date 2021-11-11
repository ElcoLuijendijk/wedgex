import numpy as np
import astropy.units as u
from shapely.geometry import LineString


def calculate_closure_temp(time, temp, ea=None, omega=None, geom=1, thermochron_system=None,
                           min_gradient=1e-7 * u.K / u.year):
    """
    calculate closure temperatures using Dodson's (1973) equations
    
    based on glide's fortran code:
    https://github.com/cirederf13/glide/blob/master/transient_geotherm/src/closure_temps.f90
    https://github.com/cirederf13/glide/blob/master/transient_geotherm/src/dodson.f90
    
    default parameters are based on Reiners and Brandon (2006, https://doi.org/10.1146/annurev.earth.34.031405.125202)
    
    """

    # populate values for ea and diff if these are not specified directly but the thermochronology system is
    if ea is None or omega is None:

        u.imperial.enable()

        # default grain size for calculating omega for the MAr thermochronometer
        a_Mar = 100.0 * 1e-6 * u.m

        # these values are taken from reiners' review paper and correspond to Ea (x1000.) and omega (x secinyr)
        if thermochron_system == 'AFT':
            # here are Ea and D0/a2 values for AFT from ketcham. 1999
            # taken from reiners 2004
            ea = 147 * 1e3 * u.J / u.mol
            geom = 1
            omega = 2.05e6 / u.s
        if thermochron_system == 'AFT_min':
            # here are Ea and D0/a2 values for AFT from ketcham. 1999
            # taken from reiners 2004
            ea = 138 * 1e3 * u.J / u.mol
            geom = 1
            omega = 5.08e5 / u.s
        if thermochron_system == 'AFT_max':
            # here are Ea and D0/a2 values for AFT from ketcham. 1999
            # taken from reiners 2004
            ea = 187 * 1e3 * u.J / u.mol
            geom = 1
            omega = 1.57e8 / u.s
        elif thermochron_system == 'ZFT':
            # here are Ea and D0/a2 values for ZFT from reiners2004
            # taken from reiners 2004
            ea = 208 * 1e3 * u.J / u.mol  # !/(4.184*1.e3)
            geom = 1
            omega = 4.0e8 / u.s
            # energy=224.d3
            # geom=1.d0
            # diff=1.24e8*3600.*24.*365.25e6
            # diff = 1.24e8*3600.*24.*365.25e6
        elif thermochron_system == 'AHe':
            # here are Ea and D0/a2 values for AHe from Farley et al. 2000
            # taken from reiners 2004
            ea = 138 * 1e3 * u.J / u.mol
            geom = 1
            omega = 7.64e7 / u.s
        elif thermochron_system == 'ZHe':
            # here are Ea and D0/a2 values for ZHe from reiners2004
            # taken from reiners 2004
            ea = 169 * 1e3 * u.J / u.mol  # !/(4.184*1.e3)
            geom = 1
            omega = 7.03e5 / u.s
            # energy=178.d3
            # geom=1.d0
            # diff=7.03d5*3600.d0*24.d0*365.25d6

        # the following are for argon argon, might be a bit much for most, 51,52,53 in glide
        elif thermochron_system == 'hbl':
            # here are Ea and D0/a2 values for hbl from harrison81
            # taken from reiners 2004
            ea = 268 * 1e3 * u.J / u.mol
            geom = 1
            omega = 1320 / u.s
        elif thermochron_system == 'MAr_old':
            # here are Ea and D0/a2 values for mus from hames&bowring1994,robbins72
            # taken from reiners 2004
            ea = 180 * 1e3 * u.J / u.mol
            geom = 1
            omega = 3.91 / u.s

        elif thermochron_system == 'MAr':
            # Values from Harrison et al. (2009, https://doi.org/10.1016/j.gca.2008.09.038)
            ea_cal = 63 * 1e3 * u.imperial.cal / u.mol
            ea = ea_cal.to(u.J / u.mol)
            geom = 1
            D0 = 2.3e-4 * u.m**2 / u.s

            omega = 55 * D0 / a_Mar**2

        elif thermochron_system == 'MAr_min':
            # Values from Harrison et al. (2009, https://doi.org/10.1016/j.gca.2008.09.038)
            ea_cal = 56 * 1e3 * u.imperial.cal / u.mol
            ea = ea_cal.to(u.J / u.mol)
            geom = 1
            D0 = 72.3 * 1e-4 * u.m**2 / u.s
            omega = 55 * D0 / a_Mar**2

        elif thermochron_system == 'MAr_max':
            # Values from Harrison et al. (2009, https://doi.org/10.1016/j.gca.2008.09.038)
            ea_cal = 70 * 1e3 * u.imperial.cal / u.mol
            ea = ea_cal.to(u.J / u.mol)
            geom = 1
            D0 = 0.1 * 1e-4 * u.m**2 / u.s
            omega = 55 * D0 / a_Mar**2

        elif thermochron_system == 'bio':
            # here are Ea and D0/a2 values for bio from grove&harrison1996
            # taken from reiners 2004
            ea = 197 * 1e3 * u.J / u.mol
            geom = 1
            omega = 733. / u.s

    R = 8.314 * u.J / (u.K * u.mol)

    cooling = np.gradient(temp, time)

    ind = cooling < min_gradient
    cooling[ind] = min_gradient

    tau = R * temp ** 2 / (ea * cooling)
    Tc = ea / (R * np.log(geom * tau * omega))

    return Tc


def calculate_closure_age(time, temp, ea=None, omega=None, geom=1, thermochron_system=None, verbose=False,
                          closure_temperature_error=0.0, subsample_T_history=5):
    """
    first calculate closure temperatures and then find the intersection between the closure temperature vs time curve 
    and the temperature vs time curve
    """

    if subsample_T_history is not None:
        # take every x temperature history points to avoid spurious changes in dT/dx with small timesteps
        # todo: replace this with a more elegant moving average algorithm
        time = time[subsample_T_history:-subsample_T_history:subsample_T_history]
        temp = temp[subsample_T_history:-subsample_T_history:subsample_T_history]

    Tc = calculate_closure_temp(time, temp, ea=ea, omega=omega, geom=geom, thermochron_system=thermochron_system)

    if closure_temperature_error is not None:
        Tc += closure_temperature_error

    if verbose is True:
        print('closure temp = ', Tc)

    if Tc.max() < temp.min() or Tc.min() > temp.max():
        # print('warning, cooling temp outside of range of thermochron temps')
        # print('range: ', Tc.min(), Tc.max())
        # raise ValueError

        return np.nan, np.nan

    else:
        xy1 = np.vstack([time.to(u.year).value, temp.value]).T
        xy2 = np.vstack([time.to(u.year).value, Tc.value]).T
        line1 = LineString(xy1)
        line2 = LineString(xy2)

        int_pt = line1.intersection(line2)
        if int_pt.type == 'MultiPoint':
            xi, yi = int_pt[0].x, int_pt[0].y
        elif int_pt.type == 'LineString':
            if len(int_pt.coords) > 0:
                # xi, yi = int_pt.coords.xy[:, 0], int_pt.coords.xy[:, 1]
                return np.nan, np.nan
            else:
                return np.nan, np.nan
        else:
            xi, yi = int_pt.x, int_pt.y

        age = xi * u.year
        Tc_int = yi * u.K

        return age, Tc_int
