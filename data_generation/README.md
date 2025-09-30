# Gyrokinetics Data Generation
Scripts and utilities for generating data with the **GKW** flux tube code.  
GKW code and documentation is available on [bitbucket](https://bitbucket.org/gkw)

## Getting Started
We provide all configuration files used to generate our dataset in the `gkw_config.zip` archive.
It contains separate directories, one per trajectory, with a single `input.dat` file in each.

After generating the trajectories with GKW, 
they can be preprocessed to the `hdf5` format with [`gyroswin.dataset.preprocess`](../gyroswin/dataset/preprocess.py).

## Parameter Ranges
We specify the ranges for sampling the main parameters:

- **`rlt`($R/L_T$, temperature gradient):** [1, 12]  
- **`rln` ($R/L_n$, density gradient):** [1, 7]  
- **`q` ($q$, safety factor):** [1, 9]  
- **`shat` ($\hat{s}$, magnetic shear):** [0.5, 5]  

Additionally, the **noise amplitude** `amp_init` of the initial condition is varied within **[1e-5, 1e-3]**.
