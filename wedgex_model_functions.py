import numpy as np
import sklearn.metrics
import wedgeqs


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


def calculate_cooling_ages(t, x_samples, d_samples, alpha, resetting_temperatures_samples, 
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
    #surface_temp_samples = surface_temperature_sea_lvl - lapse_rate * xp
    #resetting_depths = (resetting_temperatures - surface_temp_samples) / geothermal_gradient

    #target_depths_samples = resetting_depths

    n_samples = len(x_samples)

    modelled_age_samples = np.zeros((n_samples))
    
    y0_samples = x_samples * alpha
    
    for j, resetting_temperature_sample, y0s, xs, ds in zip(list(range(n_samples)), 
                                                             resetting_temperatures_samples, 
                                                             y0_samples, x_samples, d_samples):

        surface_Ts = surface_temperature_sea_lvl - y0s * lapse_rate
        
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
                                       measured_ages, age_uncertainty, resetting_temperatures_samples,
                                       default_exhumation_rate,
                                       metric_to_return,
                                       return_all=False,
                                       verbose=False):
    
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
        or 'MAE' for the mean absolute error, or 'R2' for the coefficient of determination
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
    
    #vc, vd, vxa, vya = params
    
    if 'alpha' in params_to_change:
        alpha = params[params_to_change.index('alpha')]
    if 'beta' in params_to_change:
        beta = params[params_to_change.index('beta')]
    if 'geothermal_gradient' in params_to_change:
        geothermal_gradient = params[params_to_change.index('geothermal_gradient')]
    if 'vc' in params_to_change:
        vc = params[params_to_change.index('vc')]
    if 'vd' in params_to_change:
        vd = params[params_to_change.index('vd')]
    if 'vxa' in params_to_change:
        vxa = params[params_to_change.index('vxa')]
    if 'vya' in params_to_change:
        vya = params[params_to_change.index('vya')]
    
    if limit_params is True:
        # make sure params have correct sign
        # negative for vc, vd and positive for vxa, vya
        if vc >= 0:
            vc = -1e-7
        if vd >= 0:
            vd = -1e-7
        if vxa < 0:
            vxa = 0
        if vya <0:
            vya = 0
    
    if verbose is True:
        print('parameters that were changed: ', params_to_change)
        print('parameter values: ', params)
    
    x_samples, y_samples, d_samples = run_model_multiple_samples(t, x0_samples, 
                                                                 alpha, beta, L, vc, vd, vxa, vya)
    
    y0_samples = x_samples * alpha
    
    modelled_ages = calculate_cooling_ages(t, x_samples, d_samples, alpha, 
                                           resetting_temperatures_samples, 
                                           surface_temperature_sea_lvl, 
                                           lapse_rate, geothermal_gradient,
                                           default_exhumation_rate, L)
    
    if verbose is True:
        print('modelled ages, min., mean, max.: ', 
              modelled_ages.min(), 
              modelled_ages.mean(), 
              modelled_ages.max())
    
    data = measured_ages
    prediction = modelled_ages
    
    ind_nok = (np.isnan(data)) | (np.isnan(prediction))
    
    data = data[ind_nok==False]
    prediction = prediction[ind_nok==False]
    age_und_adj = age_uncertainty[ind_nok==False]
    
    ME = np.mean(data - prediction)
    MAE = sklearn.metrics.mean_absolute_error(data, prediction)
    R2 = sklearn.metrics.r2_score(data, prediction)
    
    misfit =  misfit_function(data, age_und_adj, prediction) * len(data)
    chi_sq = np.sum(((prediction - data) / age_und_adj)**2)
    
    if verbose is True:
        print('\tME = ', ME)
        print('\tMAE = ', MAE)
        print('\tR2 = ', R2)
        print('\tmisfit = ', misfit)
        print('\tchi squared = ', chi_sq)
    
    if return_all is True:
        return (y0_samples, x_samples, y_samples, resetting_temperatures_samples, d_samples, 
                data, prediction, ME, MAE, R2)
    if metric_to_return == 'all':
        return ME, MAE, R2
    elif metric_to_return == 'R2':
        return R2
    elif metric_to_return == 'MAE':
        return MAE
    elif metric_to_return == 'misfit':
        return misfit
    elif metric_to_return == 'chisq':
        return chi_sq
    else:
        return MAE