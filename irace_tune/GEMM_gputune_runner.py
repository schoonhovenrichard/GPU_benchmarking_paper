#!/export/scratch2/schoonho/anaconda3/envs/gpubenchmarking/bin/python
##!/home/schoonho/anaconda3/envs/gpubenchmarking/bin/python
##!/usr/bin/python

#NOTE: The above is to use the correct conda isntallation!
#NOTE: Change to correct path for your installation.
#NOTE: Find path by running "which python" when the correct conda
#NOTE: Environment is active.

###############################################################################
# This script is the command that is executed every run.
# Check the examples in examples/
#
# This script is run in the execution directory (execDir, --exec-dir).
#
# PARAMETERS:
# argv[1] is the candidate configuration number
# argv[2] is the instance ID
# argv[3] is the seed
# argv[4] is the instance name
# The rest (argv[5:]) are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################

import datetime
import os.path
import re
import subprocess
import sys
import json
import numpy as np

import bloopy.utils as utils
import gpu_utils


class iRace_GPU_reader:
    def __init__(self, gpu_space):
        self.gpu_space = gpu_space
        self.fevals = 0
        self.visited = []

    def map_cfg_to_kernelcfg(self, cfg):
        param_vec = []
        for key, index in cfg:
            param_vec.append(self.gpu_space.tune_params[key[2:]][index])
        return param_vec

    def return_GPU_score(self, cfg):
        param_vec = self.map_cfg_to_kernelcfg(cfg)
        fitness = self.gpu_space.get_runtime(param_vec)
        return fitness

def compute_optimal_fitness(data):
    # Compute optimal fitness for reference
    best_fit = 100000000
    bestkey = None
    for k in data['cache'].keys():
        time = data['cache'][k]['time']
        if time < best_fit:
            best_fit = time
            bestkey = k
    print("Optimal settings in cache are:", bestkey, "with time {0:.4f}".format(best_fit))
    print("There are", len(data['cache'].keys()), "keys in the searchspace")
    return bestkey, best_fit


def target_runner_error(msg):
    # Useful function to print errors.
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)


if __name__=='__main__':
    if len(sys.argv) < 5:
        print("\nUsage: ./target-runner.py <configuration_id> <instance_id> <seed> <instance_path_name> <list of parameters>\n")
        sys.exit(1)

    # Get the parameters as command line arguments.
    configuration_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = sys.argv[3]
    instance = sys.argv[4]
    cand_params = sys.argv[5:]

    print(configuration_id, instance_id, seed, instance)

    ###  LOAD THE GPU CACHE  DATA  ###
    # Firstly, load instance to see which cache file to load.
    #with open(instance, 'r') as myfile:
    #    cache_filename = myfile.read()
    #cache_filename = cache_filename.rstrip("\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-2]) + "/"

    # Read file
    data_path = root_dir + 'GPU_benchmarking_paper/processed_cache_files/'
    #with open(data_path + cache_filename, 'r') as myfile:
    with open(instance, 'r') as myfile:
        data=myfile.read()
    data = json.loads(data)
    #print("Device: " + str(data['device_name']))
    #print("Kernel name: " + str(data['kernel_name']))
    #print("Tunable parameters: " + str(data['tune_params_keys']), end='\n\n')


    ### Pre-process the search space, remove variables with only one possible value
    searchspace_orig = data['tune_params']
    searchspace = utils.clean_up_searchspace(searchspace_orig)
    #bsize = utils.calculate_bitstring_length(searchspace)
    #print("Size of bitstring after pre-processing:", bsize)

    # Construct the GPU tuning space
    GPU_space = gpu_utils.GPU_tuning_space(searchspace, searchspace_orig, data['cache'])

    ###  CREATE iRace PARAMETERS  ###
    # Now we need to create the variables in python based on the iRace flags
    MWG = None
    NWG = None
    MDIMC = None
    NDIMC = None
    MDIMA = None
    NDIMB = None
    VWM = None
    VWN = None
    SA = None
    SB = None

    # Parse parameters
    GEMM_params = []
    print(cand_params)
    while cand_params:
        # Get and remove first and second elements.
        param = cand_params.pop(0)
        value = cand_params.pop(0)
        if param == "--MWG":
            MWG = int(value)
            GEMM_params.append((param, MWG))
        elif param == "--NWG":
            NWG = int(value)
            GEMM_params.append((param, NWG))
        elif param == "--MDIMC":
            MDIMC = int(value)
            GEMM_params.append((param, MDIMC))
        elif param == "--NDIMC":
            NDIMC = int(value)
            GEMM_params.append((param, NDIMC))
        elif param == "--MDIMA":
            MDIMA = int(value)
            GEMM_params.append((param, MDIMA))
        elif param == "--NDIMB":
            NDIMB = int(value)
            GEMM_params.append((param, NDIMB))
        elif param == "--VWM":
            VWM = int(value)
            GEMM_params.append((param, VWM))
        elif param == "--VWN":
            VWN = int(value)
            GEMM_params.append((param, VWN))
        elif param == "--SA":
            SA = int(value)
            GEMM_params.append((param, SA))
        elif param == "--SB":
            SB = int(value)
            GEMM_params.append((param, SB))
        else:
            target_runner_error("unknown parameter %s" % (param))
    print("\niRace parsed the param values:", GEMM_params)

    # Sanity checks
    if None in [MWG, NWG, MDIMC, NDIMC, MDIMA, NDIMB, VWM, VWN, SA, SB]:
        target_runner_error("Some of the variables are not set, something is wrong!")


    ###  GET CACHED FITNESS FOR GPU  ###
    irace_gpu_reader = iRace_GPU_reader(GPU_space)
    fitness = irace_gpu_reader.return_GPU_score(GEMM_params)
    print(fitness)
    raise Exception("PAUSE")
    if fitness > 1000000:
        print(str('inf') + '\n')
        sys.exit(0)
    else:
        print(str(fitness) + '\n')
        sys.exit(0)
