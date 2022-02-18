import numpy as np
import csv
import random
import json
import statistics
import math
import os

import bloopy.utils as utils
import gpu_utils

import bloopy.reproductive_functions as rep
import bloopy.selection_functions as sel
import bloopy.algorithms.genetic_algorithm as ga
import bloopy.algorithms.local_search as mls
import bloopy.algorithms.iterative_local_search as ils
import bloopy.algorithms.tabu_search as tabu
import bloopy.algorithms.genetic_local_search as gls
import bloopy.algorithms.simulated_annealing as sa
import bloopy.algorithms.differential_evolution as de
import bloopy.algorithms.hillclimbers as hill
import bloopy.algorithms.dual_annealing as dsa
import bloopy.algorithms.pso_pyswarms as psop
import bloopy.algorithms.basin_hopping as bashop
import bloopy.algorithms.random_sampling as randsl

# SMAC imports
#import logging
#logging.basicConfig(level=logging.INFO)
import ConfigSpace as CS
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.configspace import ConfigurationSpace
from smac.facade.smac_bb_facade import SMAC4BB
from smac.optimizer.acquisition import EI, PI, TS, LCB
from smac.scenario.scenario import Scenario


# Target Algorithm
class SMAC_GPU:
    def __init__(self, gpu_space):
        self.gpu_space = gpu_space
        self.fevals = 0
        self.visited = []

    def map_cfg_to_kernelcfg(self, cfg):
        param_vec = []
        for key, vals in self.gpu_space.tune_params.items():
            param_vec.append(vals[cfg[key]])
        return param_vec

    def return_GPU_score(self, cfg):
        param_vec = self.map_cfg_to_kernelcfg(cfg)
        fitness = self.gpu_space.get_runtime(param_vec)
        return fitness


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-1]) + "/"

    # Read files
    data_path = root_dir + 'GPU_benchmarking_paper/processed_cache_files/'

    convolution_files = ['convolution_A100_processed.json',
    'convolution_RTX_2070_SUPER_processed.json',
    'convolution_TITAN_RTX_processed.json',
    'convolution_V100_processed.json',
    'MI50_convolution_15x15_processed.json',
    'convolution_GTX_1080Ti_processed.json',
    'convolution_P100_processed.json',
    'convolution_K20_processed.json',
    'convolution_GTX_Titan_X_processed.json']

    GEMM_files = ['GEMM_A100_processed.json',
    'GEMM_V100_processed.json',
    'GEMM_RTX_2070_SUPER_processed.json',
    'GEMM_TITAN_RTX_processed.json',
    'MI50_GEMM_processed.json',
    'GEMM_GTX_1080Ti_processed.json',
    'GEMM_P100_processed.json',
    'GEMM_K20_processed.json',
    'GEMM_GTX_Titan_X_processed.json']

    pnpoly_files = ['pnpoly_A100_processed.json',
    'pnpoly_V100_processed.json',
    'pnpoly_RTX_2070_SUPER_processed.json',
    'pnpoly_TITAN_RTX_processed.json',
    'pnpoly_GTX_1080Ti_processed.json',
    'pnpoly_P100_processed.json',
    'pnpoly_K20_processed.json',
    'pnpoly_GTX_Titan_X_processed.json']

    #tune_files = convolution_files[1:2] + convolution_files[5:7] + GEMM_files[2:3] + GEMM_files[5:7] + pnpoly_files[2:3] + pnpoly_files[4:6]
    test_files = convolution_files[0:1] + convolution_files[2:5] + convolution_files[7:]

    for filename in test_files:
        ###  SETUP THE GPU CACHE DATA  ###

        with open(data_path + filename, 'r') as myfile:
            data=myfile.read()
        data = json.loads(data)

        print("Device: " + str(data['device_name']))
        print("Kernel name: " + str(data['kernel_name']))
        print("Tunable parameters: " + str(data['tune_params_keys']), end='\n\n')

        ### Pre-process the search space, remove variables with only one possible value
        searchspace_orig = data['tune_params']
        searchspace = utils.clean_up_searchspace(searchspace_orig)
        print("Processed search space:", searchspace)

        ### Calculate bitstring size
        bsize = utils.calculate_bitstring_length(searchspace)
        print("Size of bitstring after pre-processing:", bsize)

        ### Number of variables
        nr_vars = len(searchspace.keys())

        # Construct the GPU tuning space
        GPU_space = gpu_utils.GPU_tuning_space(searchspace, searchspace_orig, data['cache'])

        disc_space = utils.discrete_space(GPU_space.get_runtime, searchspace)

        ### Compute optimal fitness for reference
        best_fit = 100000000
        bestkey = None
        for k in data['cache'].keys():
            time = data['cache'][k]['time']
            if time < best_fit:
                best_fit = time
                bestkey = k
        print("Optimal settings in cache are:", bestkey, "with time {0:.4f}".format(best_fit))
        print("There are", len(data['cache'].keys()), "keys in the searchspace")

        ## Define experimental parameters
        #maxfevals = [25,50,75,100,150,200,400,600,800,1000,2000]
        maxfevals = [25, 50, 100, 200, 400, 800]
        minvar = 1e-10
        exper_runs = 20
        output_dir = '/experiment_files/'
        LOG_RESULTS = True


        ### SMAC SPECIFIC ###

        # Build Configuration Space which defines all parameters and their ranges.
        cs = ConfigurationSpace()
        #NOTE: SMAC does not allow for aribitrary integer distributions!
        #  Instead, we will use a uniform integer distribution over the
        # indices of the allowed values.
        cs_pars = []
        for var, vals in searchspace.items():
            param = UniformIntegerHyperparameter(var, 0, len(vals)-1, default_value=0)
            cs_pars.append(param)
        cs.add_hyperparameters(cs_pars)

        #EI, LCB, PI, TS
        acqstr = "LCB"
        # Use 'gp' or 'gp_mcmc' here
        model_type = 'gp'

        experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings","MaxFEval"]]

        # Run for different function eval limits
        for maxfeval in maxfevals:
            if acqstr == "EI":
                acq_fun = EI
            elif acqstr == "LCB":
                acq_fun = LCB
            elif acqstr == "PI":
                acq_fun = PI
            elif acqstr == "TS":
                acq_fun = TS

            # Define object for SMAC GPU tuning
            smacgpu = SMAC_GPU(GPU_space)

            # SMAC scenario object
            scenario = Scenario({
                'run_obj': 'quality',  # we optimize quality (alternative to runtime)
                'wallclock-limit': 3.5*maxfeval,  # max duration to run the optimization (in seconds)
                'cs': cs,  # configuration space
                'deterministic': 'true',
                'limit_resources': True,  # Uses pynisher to limit memory and runtime
                                          # Alternatively, you can also disable this.
                                          # Then you should handle runtime and memory yourself in the TA
                'cutoff': 10,  # runtime limit for target algorithm
                'memory_limit': 128*3072,  # adapt this to reasonable value for your hardware
                "runcount-limit": maxfeval,
            })

            results = [[],[]]
            for i in range(exper_runs):
                # Optimize, using a SMAC-object
                smac = SMAC4BB(scenario=scenario,
                               model_type=model_type,
                               #rng=np.random.RandomState(42),
                               acquisition_function=acq_fun,  # or others like PI, LCB as acquisition functions
                               tae_runner=smacgpu.return_GPU_score)

                # Start optimization
                try:
                    incumbent = smac.optimize()
                finally:
                    incumbent = smac.solver.incumbent
                #inccfg = incumbent.get_dictionary()
                #inc_vec = smacgpu.map_cfg_to_kernelcfg(inccfg)
                #inc_fitness = smacgpu.gpu_space.get_runtime(inc_vec)

                # It returns: Status, Cost, Runtime, Additional Infos
                inc_value = smac.get_tae_runner().run(
                    config=incumbent)
                #print('Optimized Value: %.4f' % inc_value[1])
                #print(smac.stats.submitted_ta_runs)
                #print(smac.stats.finished_ta_runs)
                #print(smac.stats.n_configs)
                results[0].append(best_fit/float(inc_value[1]))
                results[1].append(smac.stats.finished_ta_runs)

            settings = "model_type=" + model_type + "; acq_func=" + acqstr
            success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)

            algo_name = "SMAC4BB_"+model_type+"_"+acqstr
            experiment_results.append([algo_name, statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])

        ### Write results to file
        if LOG_RESULTS:
            export_filename = algo_name+"_"+ filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
            with open(export_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(experiment_results)
