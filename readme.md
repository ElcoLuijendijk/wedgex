# Wedgex

## Introduction

Wedgex is an analytical model of the exhumation of an orogenic wedge, and consists of two equations that describe the x and y position over time of a particle in an orogenic wedge that is transported over an inclined detachment and that undergoes internal deformation. The analytical equation can be used to calculate the time that a particle was at a certain depth and can be compared to exhumation rates inferred from thermochronological datasets such as apatite or zircon (U-Th)/He or fission track data.

The jupyter notebook [wedge_derivation_equations.ipynb](wedge_derivation_equations.ipynb) contains the derivation of the equations for the velocity and position of particles in an orogenic wedge.

The jupyter notebook [wedge_test.ipynb](wedge_test.ipynb) contains a series of tests of the analytical solutions that test the calculated particle positions to particle positions calculated using a numerical backstepping procedure. 

The jupyter notebook [wedge_model.ipynb](wedge_model.ipynb) contains a working version of the model that calculates particle trajectories and thermochronometer ages and compares these to measured thermochronometer ages. 


## Required modules

numpy, matplotlib, scipy



