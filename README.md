# Data and Plotting scripts for GPU Benchmarking 2021 paper

This repository contains the data and Python scripts for plotting the figures and generating the tables that are used in the paper about Kernel Tuner 
that is currently under submission.

The data for all experiments is stored as JSON files and sorted into subdirectories named after the kernel with which they were obtained.

## Installation

Please first ensure that the latest version of [Kernel Tuner](https://github.com/benvanwerkhoven/kernel_tuner) is installed along with all the dependencies 
required for compiling and benchmarking OpenCL and CUDA kernels. The scripts in this repository can just be executed directly without prior installation. The 
dependencies of the scripts themselves can be installed with the following command:

```
pip install -r requirements.txt

```

More information about how to install Kernel Tuner and its dependencies can be found in the [installation 
guide](http://benvanwerkhoven.github.io/kernel_tuner/install.html).


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
