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

### SMAC benchmarking

For additional benchmarking with SMAC, please also run:
```
conda install gxx_linux-64 gcc_linux-64 swig
pip install smac
```

### irace benchmarking

For additional benchmarking with irace, please install [irace](https://github.com/MLopez-Ibanez/irace) (R package). To install R, run something like:
```
sudo apt-get install r-base
```

Next, open R and run (inside R shell):
```
install.packages("irace")
```
If there are issues, please follow instructions on the [irace](https://github.com/MLopez-Ibanez/irace) page, e.g., to build from source.


## Running the experiments

For anyone interested in repeating the experiments described in the paper the following describes how to run the experiments.

The main file for running experiments is ```run_experiment.py```. It can be used as follows:

```
python run_experiment.py
```

In ```run_experiment.py```, set the bools **DETERMINISTIC** and **STOCHASTIC** to True or False depending on whether you want to run the experiments with a fixed, deterministic fitness (which will be the mean of 32 timing runs per config), or as a stochastic experiment where the fitness is a random draw from these separate timing runs.

Next, set the bool related to the algorithm you wish to use to True, and choose other parameters (such as if you want to save the results to csv, how many runs to do etc.). Also choose which GPU cache files you wish to load, by default it runs over all convolution files.

### Running SMAC experiments

To run the SMAC experiments, run

```python run_SMAC_experiments.py```

The above instructions apply here too.

### Running irace experiments

To run the irace experiments, please move to the folder

```
cd irace_tune/
```

The experiments can be run with ```run_irace_experiments.py```. Note that due to the setup, this script will be creating temporary directories to dump stdout.txt files that it reads from the shell, and it will read scenario data from certain files. To set up the experiments one needs to:

1. Change lines 29 and 60 in ```run_irace_experiments.py``` to the PATH that points to the installed irace binary.
2. Move the desired scenario file (so which kernel and GPU model you want to run) to the root directory:
```
mv scenarios/SOME_SCENARIO.txt ..
```
3. Change lines 30 and 61 to this correct scenario .txt file.
4. Make sure that the ```base_dir``` in lines 17 and 48 are empty. The script will create directories temp_dirX to save the data for each run (where X runs from ```base_dir`` up to the ```base_dir + exper_runs```), so make sure these directories don't already exist or you will overwrite results.
5. Run the code with
```
python run_irace_experiments.py
```

This will run the (stochastic) irace experiments. As mentioned in 4., the results for each run can be found in directories ```temp_dirX``` for certain numbers X. To parse all directories in one go and save the results to a .csv file run:

```
python parse_irace_data.py
```

## Re-tuning hyperparameters

For anyone interested in re-tuning the hyperparamers, one can run the tuning script as follows:

```
python tune_hyperparameters.py
```

In ```tune_hyperparameters.py```, set the bool related to the algorithm you wish to use to True, and choose other parameters (such as if you want to save the results to csv, how many runs to do etc.). Also choose which GPU cache files you wish to load, by default it runs over the P100, GTX 1080Ti, RTX 2070 Super cache files as in the paper. You can add or remove parameter values to add to the tuning run by adding the values to the ```hyperpars``` dictionary.

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

In the script select the kernel to perform competitions on by choosing the kernel string in lines 33-35. To choose mid to high-range competition, choose string from lines 37-38.

## Plot stochastic algorithm performance (Figure 2)

To obtain the algorithm performance plots from Figure 2, run:
```
python plot_stochastic_experiment.py
```

In the script select the kernel to perform competitions on by choosing the kernel string in lines 24-26.


## Plot GPU box-stripplot (Figure 3)

To create the box-stripplot of Figure 2 run:
```
python plot_gpu_minima_fitnesses.py
```

In the script, change line 58 to select another kernel than convolution.

## Plot DSA/GreedyILS per GPU (Figures 4 and 5)

To plot fraction of optimal runtime for DSA and GreedyILS per GPU, run
```
python plot_gpus.py
```

To select DSA or GreedyILS (or another algorithm) uncomment/comment lines 205-206.

## Creating and plotting FFGs (Figure 6)

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


## Plot alternative heatmaps (Figures 8 and 9)

To plot the heatmaps for different bugdet splits, run

```
python plot_algorithm_competition.py
```

To change the split, change lines 141 and 143 (they must be the same value). This represents the index in the budget list where the split is performed. For Figure 1 this is set to 4, for the 100 budget split it is set to 3, and for the 400 budget split it is set to 5. In the script select the kernel to perform competitions on by choosing the kernel string in lines 33-35. To choose mid to high-range competition, choose string from lines 37-38.

## Plot all the separate algorithm graphs (Figures 10 to 18)

To plot the algorithm performance separately for each GPU and kernel, run

```
python plot_separate_deterministic.py
```

Change the desired kernel by choosing the correct string in lines 35-37. Choose which algorithms to plot by changing line 99.

# Contributing new GPU data
**New cache files for GPUs** are always welcome! Please contact us if you generated new data and wish to share it to this GPU tuning benchmarking database. Please use the provided scripts, or new scripts with similar lay-out.

Send an email to ```richard.schoonhoven@cwi.nl```.

## Please cite us

If you use our benchmarking data, or some of our scripts in one of your research projects, please cite us:
```
@misc{TODO: WILL BE FILLED IN UPON PUBLICATION
}
```
