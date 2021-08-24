# Data and Plotting scripts for GPU Benchmarking 2021 paper

This repository contains the cached GPU tuning data, the experimental optimization algorithm data, Python scripts for re-running the experiments/tuning the GPUs, and Python scripts for plotting the figures that are used in the paper about benchmarking GPU tuning optimization algorithms that is currently under submission.

The data for all experiments is stored as JSON files where the filenames indicate the GPU model and kernel that was run. The original scripts used to generate the data can be found in ```generate_cache_scipts```. In these scripts the exact search space for the parameters can be easily found.

The raw files are stored in ```cache_files```, but the processed cache files used for experiments are stored in ```processed_cache_files```. The Python script to process cache files is ```process_cache_files.py```.

### Generating new cache files for GPUs

If users want to generate new cache scripts for different GPUs, please install the latest version of [Kernel Tuner](https://github.com/benvanwerkhoven/kernel_tuner). Running the scripts will generate new cache files for the GPU model that is being used with the same parameter search space. If users want to change the parameter search space, simply edit the correct ```kernel.py``` file and change the values for the ```tune_params``` dictionary.

## Installation

The survey and benchmarking makes use of the [BlooPy](https://github.com/schoonhovenrichard/BlooPy) Python package. Please ensure the latest version is installed.

```
pip install bloopy
```

To re-create the plots requires the seaborn package.

```
pip install seaborn
```

### Conda installation

To install and run in a (virtual) conda environment, please run:
```
conda create -n gpubenchmarking
conda activate gpubenchmarking
conda install pip #Make sure the correct pip is used, i.e., the conda version
conda install seaborn
pip install bloopy
```

## Running the experiments

For anyone interested in repeating the experiments described in the paper the following describes how to run the experiments.

The main file for running experiments is ```run_experiment.py```. It can be used as follows:

```
python run_experiment.py
```

In ```run_experiment.py```, set the bool related to the algorithm you wish to use to True, and choose other parameters (such as if you want to save the results to csv, how many runs to do etc.). Also choose which GPU cache files you wish to load, by default it runs over all convolution files.

## Re-tuning hyperparameters

For anyone interested in re-tuning the hyperparamers, one can run the tuning script as follows:

```
python tune_hyperparameters.py
```

In ```tune_hyperparameters.py```, set the bool related to the algorithm you wish to use to True, and choose other parameters (such as if you want to save the results to csv, how many runs to do etc.). Also choose which GPU cache files you wish to load, by default it runs over the P100 cache files as in the paper. You can add or remove parameter values to add to the tuning run by adding the values to the ```hyperpars``` dictionary.

After creating all the tuning files in ```tune_hyperpars_data/```, we can select the best hyperparameters by running ```choose_hyperparameters.py```

```
python choose_hyperparameters.py
```

Select different algorithms, stdev limits, and kernels to obtain optimal settings for every function evaluation bin.

# Generating the figures
## Plot Algorithm competition heatmaps (Figure 1)

To perform the statistical competition between algorithms, and plot them, run:
```
python plot_algorithm_competition.py
```

In the script select the kernel to perform competitions on. When choosing point-in-polygon, uncomment/comment lines 122-123. To choose mid to high-range competition, uncomment/comment lines 128-129.

## Plot GPU box-stripplot (Figure 2)

To create the box-stripplot of Figure 2 run:
```
python plot_gpu_minima_fitnesses.py
```

In the script, change line 58 to select another kernel than convolution.

## Plot DSA/GreedyILS per GPU (Figures 3 and 4)

To plot fraction of optimal runtime for DSA and GreedyILS per GPU, run
```
python plot_gpus.py
```

To select DSA or GreedyILS (or another algorithm) uncomment/comment lines 205-206.

## Creating and plotting FFGs (Figures 5 and 6)

To create new FFGs, run:
```
python compute_and_analyze_FFGs.py
```

Select the GPU and kernel you wish to analyze. By default, the script creates the FFG and computes the PageRank centralities (and saves them). By uncommenting line 180, the script will also draw the graph using networkX and save it as PDF. **NOTE:** Plotting FFGs for the GEMM kernels (and also most convolution kernels) is very expensive and may take a lot of RAM and time to plot.

## Plot pagerank centralities(Figure 7)

To plot FFGs proportion of PageRank centralities run:
```
python plot_centralities.py
```

In the script select the kernel to plot.

# Contributing new GPU data
**New cache files for GPUs** are always welcome! Please contact us if you generated new data and wish to share it to this GPU tuning benchmarking database. Please use the provided scripts, or new scripts with similar lay-out.

Send an email to ```richard.schoonhoven@cwi.nl```.

