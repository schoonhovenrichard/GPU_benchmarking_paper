import os
import json
from conv_gputune_runner import iRace_GPU_reader
import bloopy.utils as utils
import gpu_utils
import csv
import numpy as np
import statistics


def parse_irace_output(file):
    # Order all the entries per iRace run
    order_list = [[]]
    for line in file:
        if line[:6] == '# 2022':
            order_list.append([])
        order_list[-1].append(line)
    # Look at the final entry
    final_entry = order_list[-1]

    # Parse the actual budget used
    budget_used = None
    for ls in final_entry:
        if 'experimentsUsedSoFar' in ls:
            budget_used = int(ls.split(' ')[-1])
    if budget_used is None:
        raise Exception("Unable to parse budget used!")
    print("Parsed budget used:", budget_used)

    # Parse the best configs (can be multiple)
    configs = []
    for i in range(len(final_entry)):
        if '# Best configurations as commandlines' in final_entry[i]:
            list_of_cfgs = final_entry[i+1:]
            for entry in list_of_cfgs:
                if '   ' in entry:
                    cfg = entry.split('   ')[-1]
                elif '  ' in entry:
                    cfg = entry.split('  ')[-1]
                else:
                    raise Exception("Something wrong with parsing configs")
                cfg = cfg.split(' ')
                cfg_tup = []
                for k in range(0,len(cfg),2):
                    cfg_tup.append(tuple([cfg[k], int(cfg[k+1])]))
                configs.append(tuple(cfg_tup))
    return budget_used, configs


if __name__ == '__main__':
    #NOTE: This assumes that we currently have the experimental
    # data of one experiment in temp_dir
    rootdir = 'temp_dir'
    scenario_file = 'scenario.txt'
    with open(scenario_file) as f:
        scenario = f.readlines()
    cache_fn = ((scenario[2].split('=')[1]).split('/')[1]).split('.')[0]
    firstTest = int(((scenario[-2].split('=')[1]).strip(' ')).strip('\n'))
    nbConfigurations = int(((scenario[-1].split('=')[1]).strip(' ')).strip('\n'))
    print("PARSING iRace run:", cache_fn, "firstTest:", firstTest, "nbConfigurations:", nbConfigurations)

    # Read the original cache_file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = "/".join(current_dir.split('/')[:-2]) + "/"
    data_path = project_dir + 'GPU_benchmarking_paper/processed_cache_files/'
    with open(data_path + cache_fn + '.json', 'r') as myfile:
        data=myfile.read()
    data = json.loads(data)

    # Create the GPU space
    searchspace_orig = data['tune_params']
    searchspace = utils.clean_up_searchspace(searchspace_orig)
    GPU_space = gpu_utils.GPU_tuning_space(searchspace, searchspace_orig, data['cache'])
    irace_gpu_reader = iRace_GPU_reader(GPU_space)

    # create results dict
    budgets = [200, 400]
    result_dict = {}
    for bud in budgets:
        result_dict[bud] = {'mean_times':[], 'fevals_used':[]}

    # iterate over output files
    it = 0
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file == 'stdout.txt':
                it += 1
                # Max budget for this run
                max_budget_run = int((subdir.split('-')[-2])[1:])
                print("\nMax budget supplied", max_budget_run)

                # Parse the config results
                stdout = os.path.join(subdir, file)
                with open(stdout) as f:
                    output_file = f.readlines()
                results = parse_irace_output(output_file)
                fevals = results[0]
                best_meantime = 100000000
                best_cfg = None
                for cfg in results[1]:
                    mean_time = irace_gpu_reader.return_GPU_score(cfg)
                    if mean_time < best_meantime:
                        best_meantime = mean_time
                        best_cfg = cfg
                print(fevals, best_meantime)

                # Save to dict
                result_dict[max_budget_run]['mean_times'].append(best_meantime)
                result_dict[max_budget_run]['fevals_used'].append(fevals)


    # Save all the results to file
    exper_runs = int(it/len(budgets))
    export_filename = "tune_hyperpars_iRace_" + cache_fn + "_runs={0}".format(exper_runs) + ".csv"

    ### Compute optimal fitness for reference
    best_fit = 100000000
    bestkey = None
    for k in data['cache'].keys():
        time = data['cache'][k]['time']
        if time < best_fit:
            best_fit = time
            bestkey = k
    #print("\nOptimal settings in cache are:", bestkey, "with time {0:.4f}".format(best_fit))

    experiment_results = [[cache_fn],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

    settings = "firstTest=" + str(firstTest) + "; nbConfigurations=" + str(nbConfigurations)
    for key, vals in result_dict.items():
        times = vals['mean_times']
        fevals_used = vals['fevals_used']
        fracs = [best_fit/float(x) for x in times]
        success_rate = (np.array(fracs) == 1.0).sum()/float(exper_runs)
        experiment_results.append(["iRace", statistics.mean(fracs), statistics.stdev(fracs), success_rate, statistics.mean(fevals_used), statistics.stdev(fevals_used), settings])

    with open(export_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(experiment_results)
