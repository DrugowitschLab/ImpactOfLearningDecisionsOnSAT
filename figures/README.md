# Generating figures

These MATLAB scripts load behavioral data from the `../data` folder, and model fit and simulation data from from the `../fitdata` folder, and perform some additional analyses to generate some main text figures. The scripts will fail if the model fits and simulations have not been performed before the scripts are called.

See [../fitting/README.md](../fitting/README.md) on how to perform the model fits and simulations. See [../README.md](../README.md) for how to download pre-computed model fit and simulation data.

## Usage

Run `figureX.m` at the MATLAB command prompt to generate the panels of figure `X`. The other scripts in this folder are helper scripts that support the different `figureX.m` scripts.

To generate the model fit plots for other models, as shown in the paper supplement, modify `figure5.m` (lines 34, 105, and 184) to load the simulated behavior of the respective model.
