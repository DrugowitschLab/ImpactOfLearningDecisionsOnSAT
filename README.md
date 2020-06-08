# ImpactOfLearningDecisionsOnSAT 

This repository contains the scripts to generate the plots and fit the different models described in

André G. Mendonça, Jan Drugowitsch, M. Inês Vicente, Eric E. J. DeWitt, Alexandre Pouget, and Zachary F. Mainen (2020). [The impact of learning on perceptual decisions and its implication for speed-accuracy tradeoffs](https://doi.org/10.1038/s41467-020-16196-7). _Nature Communications_ 11:2757, 1-15.

The scripts are licensed under the [New BSD License](LICENSE).

For any questions please contact [Jan Drugowitsch](jan_drugowitsch@hms.harvard.edu).

## Preliminaries

All scripts have been run under Julia v1.0.4 on Ubuntu Linux v18.04.2 and MacOS Mojave and MATLAB 2018b under MacOS Mojave. They should also work on other operating systems or later Julia/MATLAB versions. The MATLAB scripts might also work under [GNU Octave](https://www.gnu.org/software/octave/), but we have not tested this.

Please note that this repository only contains the scripts to fit the models, generate the respective data files, and plot this data. It does not contain pre-computed parameter files, as those take a significant amount of space. Beware that fitting the models can take a consierable amount of time.

## Installation

The model fitting scripts are written in [Julia](https://julialang.org). To run these scripts, first [download a copy of Julia](https://julialang.org/downloads/) and install it. Then get a [copy of the scripts](https://github.com/DrugowitschLab/LearningPerceptualDecisions/archive/master.zip), and extract them to a folder of your choice.

To install the required Julia libraries, navigate to the `fitting` folder, and use the following commands at the Julia command-line REPL:
```
]activate .
]instantiate
```
This should download and build the required Julia libraries. An alternative approach to achieve the same is to run
```
julia --project=. -e "using Pkg; Pkg.instantiate()"
```
in a terminal in the `fitting` folder that contains the Julia scripts.

For further information about how to enter and use the Julia REPL, please consult the [Julia documentation](https://docs.julialang.org) (see standard library `REPL` section). The `]` symbol preceding the above commands initiates the REPL package manager mode. Please consult the Julia documentation (standard library `Pkg` section) for how to enter and leave the package manager mode.

Additional analysis and plotting is provided by a set of MATLAB scripts that can be found in the `figures` folder.

## Folders

Scripts and data follow the following model structure:
```
data/         Rat behavioral data
fitting/      Julia scripts to perform model fitting / simulations
fitdata/      Results of model fitting / model simulations
figures/      MATLAB scripts to generate the figures
```

## Usage

Model fitting and simulations are performed by Julia scripts in the `fitting` folder. Please consult [fitting/README.md](fitting/README.md) for details.

Additional analysis and plotting of the results are performed by the MATLAB scripts in the `figures` folder. Please consult [figures/README.md](figures/README.md) for details.

Recreating all the figures from the main paper requires extensive model fitting and simulations that might take a long, long time. For convenience, we provide the data resulting from model fits and simulations as a separate download in a [Figshare data repository](https://doi.org/10.6084/m9.figshare.12446087). To use this precomputed fits and simulations, download them from Figshare, and extract `fits.zip` into the `fitdata/figs` folder, and `modelsims.zip` into the `fitdata/modelsims` folder. This should allow you to recreate the paper figures without re-running the model fits and simulations.
