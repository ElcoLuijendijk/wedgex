#!/usr/bin/env python
# coding: utf-8

# # Wedgex
# 
# Use an analytical solution for particle trajectories in an orogenic wedge, use the particle depths to calculate thermochronometer cooling ages and compare to thermochronology data. 
# 
# 
# ## Workflow:
# 
# * This notebook was designed to calibrate the wedge exhumation model, i.e. let the model find the parameter values that provides the best fit to thermochronology data. 
# * To change the parameters to calibrate or the starting values adjust the `params_to_change` and `params` variables, see the examples below.
# 
# The default parameters and data are based on the Kuru Chu cross-section in the Himalayas, a well studied cross section which was explored Long et al. (2012, https://doi.org/10.1029/2012TC003155), Coutand et al. (2014, https://doi.org/10.1002/2013JB010891) and McQuarrie and Ehlers (2015, https://doi.org/10.1002/2014TC003783).
# 
# The thermochronology data that are used are located in the file [data/thermochron_data_projected.csv](data/thermochron_data_projected.csv). This file was created by a separate notebook [utilities/extract_xsection_data.ipynb](utilities/extract_xsection_data.ipynb), which automatically extracts thermochronology data along a cross-section.

# ## Import modules

# In[1]:


import string
import itertools
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

import sklearn.metrics

import scipy.optimize

import astropy.units as u

# equations for particle trajectories
import lib.wedgeqs as wedgeqs

# function wrappers to run multiple models and compare modelled and measured thermochron ages
import lib.wedgex_model_functions as wedgex_model_functions

# optional: advective-conductive heat flow model
import lib.heat_flow_model as hf


# In[2]:


pl.rcParams['mathtext.default'] = 'regular'


# In[3]:


try:
    from cmcrameri import cm
    cmap = cm.batlow
except:
    cmap = 'viridis'


# ## Filenames

# In[4]:


# name of file with thermochron data
thermochron_data_file = 'data/thermochron_data_projected.csv'

# column with distance
distance_column = 'projected_distance_along_xsection'
thermochronometer_col = 'system'

# filename for output file with modelled ages thermochron samples
thermochron_output_file = 'data/modelled_thermochron_data.csv'

# output file with modelled ages vs distance
thermochron_profile_file = 'data/modelled_thermochron_profiles.csv'


# ## Model parameters

# In[5]:


year = 365.25 * 24 * 3600.

model_run_name = 'default'

# compressional, transport and accretion velocity
vc = -5e-3 * u.m / u.year
vd = -5e-3 * u.m / u.year
vxa = 1e-9 * u.m / u.year
vya = 1e-9 * u.m / u.year

# convergence velocity of the downgoing plate
# note this is the horizontal component, the downgoing component is calculated usign the wedge slope
v_downgoing = 1.0e-3 * u.m / u.year

# option to calibrate parameters or to run the model with default parameters
calibrate_parameters = False

# parameters to optimize and starting values 
#model_run_name = 'calibrated'

params_to_change = ['convergence', 'conv_part', 'deform_part', 'vxa', 'vya']
#params = [20e-3 * u.m / u.year, 0.5, 0.5, 1e-5, 1e-5]

#params_to_change = ['vc', 'vd']#, 'vxa', 'vya']
#params = [2e-3 * u.m / u.year, -8e-3 * u.m / u.year]#, 1e-5 * u.m / u.year, 1e-5 * u.m / u.year]

#params_to_change = ['vc', 'vd']#, 'vxa', 'vya']
#params = [-2e-3 * u.m / u.year, -8e-3 * u.m / u.year]#, 1e-5 * u.m / u.year, 1e-5 * u.m / u.year]

#model_run_name = 'detachment_cal'
#params_to_change = ['vd']
#params = [-3e-3]

#model_run_name = 'compression_cal'
#params_to_change = ['vc']
#params = [-3e-3]

#model_run_name = 'accretion_cal'
#params_to_change = ['vxa', 'vya']
#params = [1e-5, 1e-5]


# lenght of wedge (m)
L = 200e3 * u.m

# slope of topography (m/m)
# for McQuarrie & Ehlers: ~ 4km over 117 km 
alpha = 0.034

# slope of bottom of wedge (m/m)
# for McQuarrie & Ehlers: detachment depth at N edge xsection (117 km) = ~20 km
beta = -0.17

# disctance between surface points (m)
x_interval = 5e3 * u.m

x_first_pt = 1e3 * u.m

# number of rows in the orogenic wedge. Used for figures only, no function in the actual model
n_rows = 100

# modelled timespan (years)
max_time = 2e8 * u.year

# timestep size (years)
dt = 1e4 * u.year

# thermochronology model. choose between 'simple', 'Dodson'
#thermochron_model = 'simple'
thermochron_model = 'Dodson'

# names of thermochronometers, should match the names in your input file
thermochron_systems = ['AFT', 'ZHe', 'MAr']

# only low-T thermochron
#thermochron_systems = ['AFT', 'ZHe']

# excluding low-T
#thermochron_systems = ['ZHe', 'ZFT', 'MAr']

# resetting temperatures (degr. C)
resetting_temperatures = [70.0 * u.deg_C, 110.0 * u.deg_C, 180.0 * u.deg_C, 230.0 * u.deg_C, 325.0 * u.deg_C]

# default exhumation rate, used to calculate ages from the undeformed foreland 
# that have not been reset inside the wedge (m/yr)
default_exhumation_rate = 1e-4 * u.m / u.year

# option to remove non-reset ages from database
remove_non_reset_ages = True

# limit for estimating which ages are reset or not (Ma)
reset_age_limit =  100.0 

# thermal history model. options: 'numerical' for numeircal steady-state model, 'fixed_gradient' for fixed geothermal gradient
thermal_history_model = 'numerical'

# geothermal gradient, used to convert resetting temp to depth (degr. C/m) in the case of a fixed_gradient thermal model
geothermal_gradient = 0.015 * u.deg_C / u.m

# surface temperature at sea lvl (degr. C)
surface_temperature_sea_lvl = 24.0 * u.deg_C

# adiabatic lapse rate (C/m), used to calculate surface temperatures
lapse_rate = -7.0 / 1e3 * u.deg_C / u.m

# model domain length of downgoing plate
Lxmin = 100e3 * u.m

# cellsizes
#cellsize_wedge = 2000.0 * u.m
#cellsize_footwall = 5000.0 * u.m

# cellsizes for the numerical mesh that is used for the numerical heat flow model
cellsize_wedge_top = 250.0 * u.m
cellsize_wedge_bottom = 2000.0 * u.m
cellsize_footwall = 5000.0 * u.m

# vertical size downgoing plate
Ly = 110e3 * u.m

# temperature boundary conditions
lab_temp = 1300 * u.deg_C

# thermal parameters
K = 2.5 * u.W / (u.m * u.K)
rho = 2700.0 * u.kg / u.m**3
c = 800.0 * u.J / (u.kg * u.K)

# heat prod at surface
#H0 = 2.25e-6 * u.W / u.m**3
H0 = 2.75e-6 * u.W / u.m**3

# e-folding depth heat prod
e_folding_depth = 17500.0


# In[ ]:





# ## Additional model calibration options

# In[6]:


calibration_params = {'calibration_metric': 'RMSE', 'limit_params': True, 'xtol': 1e-8, 'ftol': 1e-8}

# metric to calibrate the models. use 'MAE' or 'chisq'
calibration_metric = 'RMSE'

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
opt_method = "Nelder-Mead"

# limit parameters to realistic limits
# if True the model will limit vc <= 0, vd <= 0, vxa >= 0, vya >=0
limit_params = True

# convergence criteria for optimization algorithm
# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html for more info
xtol = 1e-8
ftol = 1e-8

# max number of iterations
maxiter = 100


# ## Set up initial particle positions and timesteps

# In[7]:


# x-coordinates of starting points:
x0s = np.arange(0, L.to(u.m).value + x_interval.to(u.m).value, x_interval.to(u.m).value) * u.m
x0s[0] = x_first_pt

# timesteps (years)
t = np.arange(0, -max_time.to(u.year).value - dt.to(u.year).value, -dt.to(u.year).value) * u.year

print('particle starting points (m from tip of wedge):\n', x0s)

print('timesteps (yr):\n', t / 1e6)


# ## Load thermochron data

# In[8]:


df = pd.read_csv(thermochron_data_file)

df[thermochronometer_col].unique()

print('thermochron data systems in file: ', df[thermochronometer_col].unique())


# ## Remove anomalously old ages

# In[9]:


if remove_non_reset_ages is True:

    print('ages before removing non-reset ages:\n', df['age'].describe())
    
    df = df[df['age'] < reset_age_limit]
    
    print('ages after removing non-reset ages:\n', df['age'].describe())


# ## Get sample data

# In[10]:


# get sample positions from dataframe
x0_samples = df[distance_column].values * u.m

# calculate surface temperature
#df['surface_T'] = surface_temperature_sea_lvl - lapse_rate * df['elevation']
#df['resetting_depth'] = (df['resetting_temp'] - df['surface_T']) / geothermal_gradient

#df.head()

measured_ages = df['age'].values * 1e6 * u.year
#data_distance = df['age'].values
measured_ages_sigma = df['age_error_1s'].values * 1e6 * u.year
#resetting_temperatures_samples = df['resetting_temp'].values

thermochron_system_samples = df[thermochronometer_col].values


# ## Parameter calibration

# In[11]:


import imp
imp.reload(wedgex_model_functions)

import warnings
warnings.filterwarnings('ignore')

params_ = []

for p in params:
    if type(p) == u.quantity.Quantity:
        params_.append(p.value)
    else:
        params_.append(p)
        
#params_ = [p.value for p in params]
#params_ = params

args = (params_to_change, limit_params, t, x0_samples, alpha, beta, L, vc, vd, vxa, vya, 
        surface_temperature_sea_lvl, lapse_rate, geothermal_gradient,
        measured_ages, measured_ages_sigma, default_exhumation_rate,
        calibration_metric, 
        thermal_history_model, Ly, Lxmin, cellsize_wedge_top, cellsize_wedge_bottom, cellsize_footwall, 
        lab_temp, K, rho, c, H0, e_folding_depth, v_downgoing, thermochron_model, thermochron_systems, thermochron_system_samples,
        resetting_temperatures)

# test run
#init_results = wedgex_model_functions.compare_modelled_and_measured_ages(params_, *args)

print('using %s metric to calibrate model ' % calibration_metric)

print('starting optimization, this may take a while, especially in combination with the numerical heat flow model ....')
#opt_results = scipy.optimize.fmin(wedgex_model_functions.compare_modelled_and_measured_ages, 
#                                  params_, args=args, 
#                                  maxiter=maxiter, xtol=xtol, ftol=ftol, full_output=True)
opt_options = {"full_output": True, "maxiter": maxiter}

opt_results = scipy.optimize.minimize(wedgex_model_functions.compare_modelled_and_measured_ages, 
                                      params_, args=args, method=opt_method, options=opt_options)

print('done optimizing')


# In[ ]:


params = opt_results.x
opt_error = opt_results.fun

print('optimized parameter values ', params)
print('optimized model error ', opt_error)


# ## Save calibration results 

# In[ ]:


param_tries, model_errors = opt_results.final_simplex

dfr = pd.DataFrame(index=np.arange(len(model_errors)), columns=params_to_change + ['model_error'])

for i, p in enumerate(params_to_change):
    dfr[p] = param_tries[:, i]
    
dfr["model_error"] = model_errors

today = datetime.datetime.now()
today_str = '%i-%i-%i' % (today.day, today.month, today.year)
fn = f"data/calibrated_parameters_{len(model_errors)}_tries_{today_str}.csv"
dfr.to_csv(fn, index_label="calibrated_parameter")

dfr


# ## Save input data as pickle file

# In[ ]:


fnp = f"data/calibrated_parameters_{len(model_errors)}_tries_{today_str}.pck"
fout = open(fnp, "wb")
pickle.dump(args, fout)
fout.close()


# In[ ]:


lapse_rate, c


# In[ ]:




