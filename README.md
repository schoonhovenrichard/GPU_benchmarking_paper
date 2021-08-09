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

The main file for running experiments is experiment.py. It can be used as follows:

```
./experiment.py [algorithm] [strategy]
```

Where algorithm is any of 'convolution', 'gemm', 'gemm_amd', or 'pnpoly', and strategy is any of 'brute_force', 'minimize', 'basinhopping', 'diff_evo'.

The command will start to benchmark the algorithm on the current device using the specified strategy. All supported methods for that strategy will be 
benchmarked, and every run will be performed 7 times, except for when 'brute_force' is used. Note that all kernel execution times are also measured as the
average of 7 kernel executions by default.

experiment.py will store the tuning results using the JSON format in the algorithm's subdirectory. Also note that experiment.py will continue where it left off 
if for some reason the previous run did not run to completion, for example when the compute node reservation time ended.

If you wish to add a kernel to this experiment, please extend the algorithm dictionary inside experiment.py with another entry.


## Generating the violin plots

```
./violins.py [algorithm]
```

When all data for the experiments of the specified algorithm is present violins.py will generate three different violin plots, one for each of the strategies.
The plots will be stored as PDF and PNG using the filename format [algorithm]-[strategy].[pdf/png]. Also, a matplotlib pop-up will be shown
for each generated plot.

## Generating the scatter plots

```
./scatter.py [algorithm]
```

scatter.py will generate the scatter plots that plot the performance of the best performing kernel configuration against the performance of the tuner using a 
particular strategy. The plots will be stored as PDF and PNG using the filename format [algorithm]-summary.[pdf/png]. Also, a matplotlib pop-up will be shown 
with the plot. Information used for the tables that show averages and standard deviations are also printed to standard output, in a format ready to be 
copy/pasted into a LaTeX tabular environment.

# Contributing new GPU data
**New cache files for GPUs** are always welcome! Please contact us if you generated new data and wish to share it to this GPU tuning benchmarking database. Please use the provided scripts, or new scripts with similar lay-out.


