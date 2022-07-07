# Precond-Cahn-Hilliard-OSC

This repository contains python3 codes reproducing the results from the 

**Paper:**
[1] Preconditioning for a phase-field model with application to morphology evolution in organic semiconductors. K. Bergermann, C. Deibel, R. Herzog, R. C. I. MacKenzie, J.-F. Pietschmann and M. Stoll. arXiv:2204.03575. 2022.

**Requirements:**
 - numpy (tested with version 1.21.5)
 - scipy (tested with version 1.6.0)
 - dolfinx, ufl (tested with [docker](https://hub.docker.com/r/dolfinx/dolfinx) version dolfinx/dolfinx:latest as of 2022-02-16)
 - mpi4py (tested with version 3.1.3)
 - petsc4py (tested with version 3.16.4)
 - pyamg (tested with version 4.2.3)

**Version:**
All codes were tested with python 3.9.7 on Ubuntu 20.04.4 LTS.

**Visualization:**
The simulation results generate .h5 and .xdmf output files. The .xdmf files can be visualized using Paraview (tested with version 5.7.0).


This repository contains:

**License:**
 - LICENSE: GNU General Public License v2.0
 
**Directory:**
 - results: collects output files of simulations
 
**Main script:**
 - Precond_CH_OSC_morphologies.py: contains the implementation of the model, discretization, preconditioning, and solution of the preconditioned system from [1].
