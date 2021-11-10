import sys
import itertools
import numpy as np
import sklearn.metrics
import lib.wedgeqs as wedgeqs

import scipy.interpolate

import shapely
from shapely.geometry import LineString, Point

import astropy.units as u

# optional: advective-conductive heat flow model
import lib.heat_flow_model as hf


import lib.thermochron_functions as thermochron_functions

#thermochron_functions


def calculate_closure_age_simple(time, temp, resetting_temp):
    
    """
    calculate thermochronometer closure age using a fixed closure temperature
    """
    
    #ind = np.isnan(temp==False)
    
    Tr = resetting_temp.to(u.K, equivalencies=u.temperature())

    age = np.interp([Tr], temp, time, left=np.nan, right=np.nan)[0]
    
    #try:
    #    age = np.interp([Tr], temp, time, left=np.nan, right=np.nan)[0]
    #except:
    #    print([Tr], temp, time)
    #    raise ValueError
        
    return age
    
    
def misfit_function(M, sigma, C):
    
    """
    misfit function following PECUBE
    """
    
    n = len(M)
    
    misfit = 1.0 / n * np.sqrt(np.sum((M - C)**2 / (sigma**2)))
    
    return misfit
    

def run_model_multiple_samples(t, x0s, alpha, beta, L, vc, vd, vxa, vya):
    
    """
    Calculate particles trajectories for multiple samples / starting points
    and calculate particle depths

    Parameters
    ----------
    t : array-like
        Time (years)
    x0s : array-like
        x-coordinate of the starting points of the particles. Note that the y coordinate is assumed to be at the land surface
    alpha : float
        slope of the surface (m/m)
    beta : float
        slope of the base of the wedge (m/m)
    L : float
        Length of the wedge (m)
    vc : float
        Compression velocity (m/yr)
    vd : float
        Horizontal component of transport velocity along the basal detachment (m/yr)
    vxa : float
        Horizontal transport velocity due to accretion (m/yr)
    vya : float
        Vertical transport velocity due to accretion (m/yr)

    Returns
    -------
    xp : array-like
        X coordinates of the particle trajectories
    yp : array-like
        Y coordinates of the particle trajectories
    dp : array-like
        Particle depths

    """

    # analytical solution for particle trajectories
    xyp = np.array([wedgeqs.analytical_solution(t, x0, alpha, beta, L, vc, vd, vxa, vya)
                    for x0 in x0s])

    # get x and y coordinates particles
    xp, yp = xyp[:, 0], xyp[:, 1]

    # calculate depth of the particles
    yp0 = xp * alpha
    dp = yp0 - yp
    
    return xp, yp, dp


def interpolate_thermal_history(x_samples, y_samples, Tx, Ty, T, remove_T_jumps=True, max_T_change=5.0*u.deg_C):
    
    T_history_samples = np.zeros(x_samples.shape) * u.deg_C
    
    Txy = np.vstack([Tx, Ty]).T
    
    # set up interpolator
    T_int = scipy.interpolate.LinearNDInterpolator(Txy, T) #,rescale=False) 
    
    #xy_samples = np.vstack([x_samples, y_samples])
    
    # interpolate to get T values for samples from modelled T mesh
    for i, xsi, ysi in zip(itertools.count(), x_samples, y_samples):
        xysi = np.vstack([xsi, ysi]).T
        
        ind_ok = np.any(np.isnan(xysi) == False, axis=1)
        
        T_history_samples[i][ind_ok] = T_int(xysi[ind_ok]) * u.deg_C
        T_history_samples[i][ind_ok==False] = np.nan
        
    # remove values with high 
    if remove_T_jumps == True:
        for i, Th in enumerate(T_history_samples):
            d = np.abs(np.diff(Th))
            ind_nok = d > max_T_change
            T_history_samples[i][1:][ind_nok] = np.nan
            
    return T_history_samples


def calculate_cooling_ages_simple(t, x_samples, d_samples, resetting_temperatures_samples, T_history_samples,
                                   surface_temperature_sea_lvl, lapse_rate, geothermal_gradient,
                                   default_exhumation_rate, L,
                                   calculate_non_reset_age=True, buffer=0.99,
                                   verbose=False, debug=True):
    
    
    n_samples = len(d_samples)
    calculate_non_reset_age = True
    buffer = 0.99

    modelled_age_samples = np.zeros((n_samples)) * u.year

    for j, resetting_temperature_sample,  xs, ds, T_history_sample in zip(list(range(n_samples)), 
                                                                 resetting_temperatures_samples, 
                                                                 x_samples, d_samples, T_history_samples):


        # find indices of steps when particle is inside the model domain
        ind = (np.isnan(ds) == False) & (np.isnan(T_history_sample) == False)

        # if the particle is in the model domain at least part of the time
        if np.any(ind):
            
            #T_history_sample = surface_Ts + ds * geothermal_gradient

            # get age of sample for section in the model domain
            # if max temp < resetting temp
            if T_history_sample[ind].max() > resetting_temperature_sample:
                age_int = np.interp(resetting_temperature_sample, T_history_sample[ind], t[ind]) 
                
            # otherwise assume that the sample was at the same temp as the last 
            elif calculate_non_reset_age is True:

                ind2 = np.isnan(xs) == False
                #if np.any(ind2):
                #    print(xs[ind2].max())

                # todo: figure out if sample comes from foreland or hinterland
                #buffer = 0.99
                if np.any(xs >= (L * buffer)):
                    from_foreland = False
                    age_int = np.nan
                else:

                    age_int = - (resetting_temperature_sample - surface_temperature_sea_lvl) \
                                / (default_exhumation_rate * geothermal_gradient)
            else:
                age_int = np.nan

        elif calculate_non_reset_age is True:

            ind2 = np.isnan(xs) == False
            if np.any(ind2):
                print(xs[ind2].max())

            # todo: figure out if sample comes from foreland or hinterland
            if np.any(ind2) and np.max(xs[ind2]) >= L:
                from_foreland == False
                age_int = np.nan

            else:

                age_int = - (resetting_temperature_sample - surface_temperature_sea_lvl) \
                                / (default_exhumation_rate * geothermal_gradient)
        else:
            age_int = np.nan

        if np.isnan(age_int) == False:
            try:
                modelled_age_samples[j] = -age_int / 1e6
            except:
                print(modelled_age_samples[j], age_int, default_exhumation_rate, geothermal_gradient)
        else:
            modelled_age_samples[j] = np.nan
        
    return modelled_age_samples
        

def calculate_cooling_ages_old(t, x_samples, d_samples, alpha, resetting_temperatures_samples, 
                           surface_temperature_sea_lvl, lapse_rate, geothermal_gradient,
                           default_exhumation_rate, L,
                           calculate_non_reset_age=True, buffer=0.99,
                           verbose=False, debug=True):
    
    """
    Calculate cooling ages for samples


    Parameters
    ----------
    t : array-like
        Time (yr)
    x_samples  : array-like
        x-coordinates of particle trajectories of samples
    d_samples : array-like
        depth of particle trajectories for samples
    alpha : float
        slope of the land surface
    resetting_temperatures_samples : array-like
        resetting temperature of each sample
    surface_temperature_sea_lvl : float
        surface temperature at sea level (degr. C)
    lapse_rate : float
        atmospheric lapse rate (degr. C / m)
    geothermal_gradient : float
        geothermal gradient (degr. C / m)
    calculate_non_reset_age : boolean
        Estimate an age for samples that have not reached their resetting temperature inside
        the wedge. Default=True
    default_exhumation_rate: float
        Default exhumation rate to calculate the non-reset age
    buffer: float
        buffer for checking if samples originate in the hinterland or not. all samples with the first point at x >= L * buffer are considered to have originated in the hinterland, all others in the foreland. This is a bit hackish, will be replaced by something more elegant at some stage.
    verbose : boolean
        extra output
    debug : boolean
        Flag for debugging the code

    Returns
    -------
    modelled_age_samples : array-like
        Modelled thermochronometer ages (Ma)
    """
    
    # calculate surface temperature
    #surface_temp_samples = surface_temperature_sea_lvl + lapse_rate * xp
    #resetting_depths = (resetting_temperatures - surface_temp_samples) / geothermal_gradient

    #target_depths_samples = resetting_depths

    n_samples = len(x_samples)

    modelled_age_samples = np.zeros((n_samples))
    
    y0_samples = x_samples * alpha
    
    for j, resetting_temperature_sample, y0s, xs, ds in zip(list(range(n_samples)), 
                                                             resetting_temperatures_samples, 
                                                             y0_samples, x_samples, d_samples):

        surface_Ts = surface_temperature_sea_lvl + y0s * lapse_rate
        
        ind = np.isnan(ds) == False
        
        if np.any(ind):
        
            T_history_sample = surface_Ts + ds * geothermal_gradient

            if T_history_sample[ind].max() > resetting_temperature_sample:
                age_int = np.interp(resetting_temperature_sample, T_history_sample[ind], t[ind])
            elif calculate_non_reset_age is True:

                ind2 = np.isnan(xs) == False
                #if np.any(ind2):
                #    print(xs[ind2].max())
                
                # todo: figure out if sample comes from foreland or hinterland
                #buffer = 0.99
                if np.any(xs >= (L * buffer)):
                    from_foreland = False
                    age_int = np.nan
                else:
                
                    age_int = - (resetting_temperature_sample - surface_temperature_sea_lvl) \
                                / (default_exhumation_rate * geothermal_gradient)
            else:
                age_int = np.nan
        
        elif calculate_non_reset_age is True:
            
            ind2 = np.isnan(xs) == False
            if np.any(ind2):
                print(xs[ind2].max())
            
            # todo: figure out if sample comes from foreland or hinterland
            if np.any(ind2) and np.max(xs[ind2]) >= L:
                from_foreland == False
                age_int = np.nan
                
            else:
                
                age_int = - (resetting_temperature_sample - surface_temperature_sea_lvl) \
                                / (default_exhumation_rate * geothermal_gradient)
        else:
            age_int = np.nan
            
        if np.isnan(age_int) == False:
            modelled_age_samples[j] = -age_int / 1e6
        else:
            modelled_age_samples[j] = np.nan
         
    return modelled_age_samples


def compare_modelled_and_measured_ages(params, params_to_change, limit_params, t, x0_samples, 
                                       alpha, beta, L, vc, vd, vxa, vya, 
                                       surface_temperature_sea_lvl, lapse_rate, geothermal_gradient,
                                       measured_ages, age_uncertainty,
                                       default_exhumation_rate,
                                       metric_to_return,
                                       thermal_history_model,
                                       Ly, Lxmin, cellsize_wedge_top, cellsize_wedge_bottom, cellsize_footwall, 
                                       lab_temp, K, rho, c, H0, e_folding_depth, v_downgoing,
                                       thermochron_model, thermochron_systems, thermochron_system_samples,
                                       resetting_temperatures,
                                       return_all=False,
                                       verbose=True):
    
    """
    Calculate particle trajectories and cooling ages and compare these to measured ages

    Parameters
    ----------
    params : list
        parameter values for this particular model run
    params_to_change : list
        List of names of parameters to change
    limit_params : boolean
        Option to limit parameters to within realistic ranges. vc<=0, vd<=0, vxa>=0, vya>=0
    t : array-like
        Time (yr)
    x0_samples  : array-like
        x-coordinates of starting points of samples (m)
    alpha : float
        slope of the surface (m/m)
    beta : float
        slope of the base of the wedge (m/m)
    L : float
        Length of the wedge (m)
    vc : float
        Compression velocity (m/yr)
    vd : float
        Horizontal component of transport velocity along the basal detachment (m/yr)
    vxa : float
        Horizontal transport velocity due to accretion (m/yr)
    vya : float
        Vertical transport velocity due to accretion (m/yr)
    surface_temperature_sea_lvl : float
        surface temperature at sea level (degr. C)
    lapse_rate : float
        atmospheric lapse rate (degr. C / m)
    geothermal_gradient : float
        geothermal gradient (degr. C / m)
    measured_ages : array-like
        Measured thermochronometer ages (Ma)
    age_uncertainty : array-like
        Uncertainty of measured thermochronometer ages (Ma)
    resetting_temperatures_samples : array-like
        Resetting temperature of each sample
    default_exhumation_rate : float
        default exhumation rate, used to calculate the age for samples that have not been reset inside
        the wedge
    weigh_residuals : boolean
        option to weight the error, by dividing the model error with the uncertainty
    metric_to_return : string
        option for the error metric to return. 'all' for ME, MAE, R2,
        or 'MAE' for the mean absolute error, 'R2' for the coefficient of determination, 'RMSE' for root mean squared error
    return_all : boolean
        full output, containing the following variables: (y0_samples, x_samples, y_samples,
        resetting_temperatures_samples, d_samples, data, prediction, ME, MAE, R2)
    verbose : boolean
        more extensive screen output

    Returns
    -------
    MAE : float
        Mean absolute error modelled and measured ages (Ma)
    """
    
    year = 365.25 * 24 * 3600.
    
    #vc, vd, vxa, vya = params
    
    if 'convergence' in params_to_change:
        conv = params[params_to_change.index('convergence')] * u.m / u.year
        conv_part = params[params_to_change.index('conv_part')]
        def_part = params[params_to_change.index('deform_part')]
        
        v_downgoing = conv * conv_part 
        v_wedge = - (1.0 - conv_part) * conv
        vc = def_part * v_wedge
        vd = (1.0 - def_part) * v_wedge
    
    if 'alpha' in params_to_change:
        alpha = params[params_to_change.index('alpha')]
    if 'beta' in params_to_change:
        beta = params[params_to_change.index('beta')]
    if 'geothermal_gradient' in params_to_change:
        geothermal_gradient = params[params_to_change.index('geothermal_gradient')]
    if 'vc' in params_to_change:
        vc = params[params_to_change.index('vc')] * u.m / u.year
    if 'vd' in params_to_change:
        vd = params[params_to_change.index('vd')]  * u.m / u.year
    if 'vxa' in params_to_change:
        vxa = params[params_to_change.index('vxa')]  * u.m / u.year
    if 'vya' in params_to_change:
        vya = params[params_to_change.index('vya')]  * u.m / u.year
    
    if limit_params is True:
        # make sure params have correct sign
        # negative for vc, vd and positive for vxa, vya
        if vc >= 0:
            vc = -1e-7 * u.m / u.year
        if vd >= 0:
            vd = -1e-7 * u.m / u.year
        if vxa < 0:
            vxa = 0 * u.m / u.year
        if vya < 0:
            vya = 0 * u.m / u.year
    
    if verbose is True:
        print('parameters that were changed: ', params_to_change)
        print('parameter values: ', params)
    
    x_samples_, y_samples_, d_samples_ = run_model_multiple_samples(t, x0_samples, 
                                                                 alpha, beta, L, vc, vd, vxa, vya)
    
    x_samples, y_samples, d_samples = x_samples_* u.m, y_samples_* u.m, d_samples_ * u.m
    #print(t, x0_samples, alpha, beta, L, vc, vd, vxa, vya)
    #print(x_samples, y_samples)
    #print(bla)
    
    y0_samples = x_samples * alpha
    
    
    # remove dimensions
    L_, Ly_, Lxmin_, cellsize_wedge_top_, cellsize_wedge_bottom_, cellsize_footwall_ = \
    L.to(u.m).value, Ly.to(u.m).value, Lxmin.to(u.m).value, cellsize_wedge_top.to(u.m).value, cellsize_wedge_bottom.to(u.m).value, cellsize_footwall.to(u.m).value

    x0_samples_ = x0_samples.to(u.m).value

    vd_, vc_, vxa_, vya_, v_downgoing_ = \
    vd.to(u.m/u.s).value, vc.to(u.m/u.s).value, vxa.to(u.m/u.s).value, vya.to(u.m/u.s).value, v_downgoing.to(u.m/u.s).value

    t_ = t.to(u.s).value
    #
    #Tx, Ty, T = hf.model_heat_transport(L, Ly, alpha, beta, Lxmin, cellsize_wedge, cellsize_footwall, 
    #                                    vd / year, vc / year, vxa / year, vya / year, v_downgoing, surface_temperature_sea_lvl, 
    #                                    lapse_rate, lab_temp, 
    #                                    K, rho, c, H0, e_folding_depth)
    
    if thermal_history_model == 'numerical':
    
        Tx_, Ty_, T_, q, mesh = hf.model_heat_transport(L_, Ly_, alpha, beta, Lxmin_, 
                                                        cellsize_wedge_top_, cellsize_wedge_bottom_, cellsize_footwall_, 
                                                        vd_, vc_, vxa_, vya_, v_downgoing_, surface_temperature_sea_lvl.value, 
                                                        lapse_rate.value, lab_temp.value, 
                                                        K.value, rho.value, c.value, H0.value, e_folding_depth)
        
        #Lx, Ly, alpha, beta, Lxmin, cellsize_wedge_top, cellsize_wedge_bottom, cellsize_footwall, 
         #                vd, vc, vxa, vya, v_downgoing, 
         #               sea_lvl_temp, lapse_rate, lab_temp, K, rho, c, H0, e_folding_depth
        print("modelled temps ", T_.mean(), T_.min(), T_.max())
        
        Tx, Ty, T = Tx_ * u.m, Ty_ * u.m, T_ * u.deg_C

    #print('numerical heat transport model done')
    #sys.stdout.write("\r" + str(f))
    sys.stdout.write('.')
    sys.stdout.flush()
    
    #modelled_ages = calculate_cooling_ages(t, x_samples, d_samples, alpha, 
    #                                       resetting_temperatures_samples, 
    #                                       surface_temperature_sea_lvl, 
    #                                       lapse_rate, geothermal_gradient,
    #                                       default_exhumation_rate, L)
    
    # get temp history of samples
    if thermal_history_model is 'numerical':
        T_history_samples = interpolate_thermal_history(x_samples, y_samples, Tx, Ty, T)
    else:
        n_samples = len(x_samples)

        #modelled_age_samples = np.zeros((n_samples))
        T_history_samples = np.zeros(x_samples.shape) * u.deg_C

        y0_samples = x_samples * alpha

        for j, y0s, xs, ds in zip(list(range(n_samples)), y0_samples, x_samples, d_samples):

            surface_Ts = surface_temperature_sea_lvl + y0s * lapse_rate

            ind = np.isnan(ds) == False

            if np.any(ind):
                T_history_samples[j] = surface_Ts + ds * geothermal_gradient
    
    n_samples = len(x_samples)
    modelled_ages = np.zeros((n_samples)) * u.year

    for i, tcs in enumerate(thermochron_systems):

        inds = thermochron_system_samples == tcs

        #print(np.sum(inds))

        for j in range(n_samples):
            if inds[j] == True:
                #print('ok ',j)
                Tpi = T_history_samples[j].to(u.K, equivalencies=u.temperature()) 
                ind_ok = np.isnan(Tpi) == False

                ti = -t[ind_ok]

                #print(len(Tpi), np.sum(ind_ok))

                if np.sum(ind_ok) > 1:

                    #print(t[ind_ok], Tpi[ind_ok], tcs)
                    if thermochron_model == 'simple':
                        modelled_ages[j] = calculate_closure_age_simple(-t[ind_ok], Tpi[ind_ok], resetting_temperatures[i])
                    else:
                        #modelled_ages_samples[j], ct = wedgex_model_functions.calculate_closure_age(-t[ind], Tpi[ind], tcs)

                        #modelled_ages[j] = calculate_closure_age(ti.to(u.s), Tpi[ind_ok], tcs)
                            
                        #modelled_ages[j] = calculate_closure_age(ti, Tpi[ind_ok], tcs)
                        modelled_ages[j], ct =  thermochron_functions.calculate_closure_age(ti, Tpi[ind_ok], thermochron_system=tcs)
                        
                else:
                    #print(Tpi)
                    #print(bla)
                    pass
            else:
                pass
                        
    if verbose is True:
        ma = modelled_ages[np.isnan(modelled_ages)==False]
        print("modelled ages, min., mean, max.: ", 
              ma.min(), 
              ma.mean(), 
              ma.max())

    data = measured_ages
    prediction = modelled_ages
    
    ind_nok = (np.isnan(data)) | (np.isnan(prediction))
    
    data = data[ind_nok==False]
    prediction = prediction[ind_nok==False]
    age_und_adj = age_uncertainty[ind_nok==False]
    
    ME = np.mean(data - prediction)
    MAE = sklearn.metrics.mean_absolute_error(data, prediction)
    RMSE = sklearn.metrics.mean_squared_error(data, prediction, squared=False)
    R2 = sklearn.metrics.r2_score(data, prediction)
    
    misfit =  misfit_function(data, age_und_adj, prediction) * len(data)
    chi_sq = np.sum(((prediction - data) / age_und_adj)**2)
    
    if verbose is True:
        print("\tME = ", ME / 1e6)
        print('\tMAE = ', MAE / 1e6)
        print('\tRMSE = ', RMSE / 1e6)
        print('\tR2 = ', R2)
        print('\tmisfit = ', misfit)
        print('\tchi squared = ', chi_sq)
        print('-----')
    
    if return_all is True:
        return (y0_samples, x_samples, y_samples, d_samples, 
                data, prediction, ME, MAE, R2)
    if metric_to_return == 'all':
        return ME, MAE, R2
    elif metric_to_return == 'R2':
        return R2
    elif metric_to_return == 'MAE':
        return MAE
    elif metric_to_return == 'RMSE':
        return RMSE
    elif metric_to_return == 'misfit':
        return misfit
    elif metric_to_return == 'chisq':
        return chi_sq
    else:
        return MAE