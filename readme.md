# Wedgex

## Introduction

Wedgex is an analytical model of the exhumation of an orogenic wedge, and consists of two equations that describe the x and y position over time of a particle in an orogenic wedge that is transported over an inclined detachment and that undergoes internal deformation. The analytical equation can be used to calculate the time that a particle was at a certain depth and can be compared to exhumation rates inferred from thermochronological datasets such as apatite or zircon (U-Th)/He or fission track data.

The jupyter notebook [wedgex_derivation_equations.ipynb](wedgex_derivation_equations.ipynb) contains the derivation of the equations for the velocity and position of particles in an orogenic wedge.

The jupyter notebook [wedgex_test.ipynb](wedgex_test.ipynb) contains a series of tests of the analytical solutions that test the calculated particle positions to particle positions calculated using a numerical backstepping procedure. 

The jupyter notebook [wedgex_model.ipynb](wedgex_model.ipynb) contains a working version of the model that calculates particle trajectories and thermochronometer ages and compares these to measured thermochronometer ages. 


## Required Python modules

Wedgex requires the following Python modules:
[numpy](https://numpy.org/), [matplotlib](https://matplotlib.org/), [pandas](https://pandas.pydata.org/)

In addition [Jupyter](https://jupyter.org/) needs to be installed to be able to run the notebooks.

Note that all of these modules and Jupyter are included in a Python distribution such as [Anaconda](https://www.anaconda.com/distribution/)





