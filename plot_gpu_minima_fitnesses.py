import numpy as np
import json
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from bloopy.individual import individual, continuous_individual

import bloopy.utils as utils
import bloopy.analysis.analysis_utils as anutil
import bloopy.analysis.critical_points as crit_util
import gpu_utils

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-1]) + "/"

    # Read file
    # We do hyperparameter tuning on GTX 1080Ti files
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

    GPUs = ["A100", "V100", "RTX_2070_SUPER", "TITAN_RTX", "P100", "K20", "GTX_Titan_X", "GTX_1080Ti", "MI50"]
    pnpoly_GPUs = ["A100", "V100", "RTX_2070_SUPER", "TITAN_RTX", "P100", "K20", "GTX_Titan_X", "GTX_1080Ti"]
    dat = [None,None,None,None,None,None,None,None,None]

    data_lst = []
    data_lst2 = []
    for filename in convolution_files:
        with open(data_path + filename, 'r') as myfile:
            data=myfile.read()
        data = json.loads(data)

        #print("Device: " + str(data['device_name']))
        #print("Kernel name: " + str(data['kernel_name']))
        #print("Tunable parameters: " + str(data['tune_params_keys']), end='\n\n')

        # Pre-process the search space
        searchspace_orig = data['tune_params']
        searchspace = utils.clean_up_searchspace(searchspace_orig)
        #print("Processed search space:", searchspace)

        ### Calculate bitstring size
        bsize = utils.calculate_bitstring_length(searchspace)
        #print("Size of bitstring after pre-processing:", bsize)

        ### Number of variables
        nr_vars = len(searchspace.keys())

        # Construct the GPU tuning space
        GPU_space = gpu_utils.GPU_tuning_space(searchspace, searchspace_orig, data['cache'])
        disc_space = utils.discrete_space(GPU_space.get_runtime, searchspace)

        ### Compute optimal fitness for reference
        best_fit = 100000000
        #best_fit = 0
        bestkey = None
        for k in data['cache'].keys():
            #time = data['cache'][k].get('MPoints/s', 0)
            time = data['cache'][k]['time']
            if time < best_fit:
            #if time > best_fit:
                best_fit = time
                bestkey = k
        print("Optimal settings in cache are:", bestkey, "with time {0:.4f}".format(best_fit))
        #print("There are", len(data['cache'].keys()), "keys in the searchspace")

        ###  <<<  ANALYZING SEARCH SPACES  >>>
        #method = 'Hamming'
        method = 'bounded'
        boundary_list = utils.generate_boundary_list(searchspace)
        best_bit = gpu_utils.convert_gpusetting_to_bitidxs(bestkey, boundary_list, searchspace_orig)
        indiv = individual(bsize, boundary_list=boundary_list)
        utils.set_bitstring(indiv, list(best_bit))
        glob_fit = disc_space.fitness(indiv.bitstring)
        print("Global minimum:", bestkey, "with fitness", glob_fit)

        nidxs_dict = anutil.build_nodeidxs_dict(boundary_list, disc_space.fitness, bsize)
        tot, minimas, maximas, saddles, regulars, spacedict = crit_util.classify_points(bsize, boundary_list, nidxs_dict, method=method)
        _, _, depths = crit_util.sizes_minima(searchspace, bsize, boundary_list, disc_space.fitness, method=method)

        print("\n", filename)
        #print("Optimal settings in have fitness:", best_fit)
        #print("Number of minima", minimas, "out of", tot)
        frac_minima = minimas/float(tot)

        fgarr = np.array(depths[:,0]).copy()
        fgarr = best_fit / fgarr
        frac_global = np.mean(fgarr)
        #print("Average fraction of global optimum:", frac_global)

        gpu = None
        for gp in GPUs:
            if gp in filename:
                gpu = gp
        if gpu is None:
            raise Exception("Unknown GPU")

        data_lst.append([gpu, frac_global, frac_minima])
        for x in depths[:,0]:
            data_lst2.append([gpu, best_fit / x])

    columns = ["GPU", "Frac_global", "Frac_minima"]
    datadf = pd.DataFrame(data_lst, columns=columns)
    datadf = datadf.sort_values('Frac_global', ascending=False)

    columns2 = ["GPU", "Fitness_minima"]
    fitdf = pd.DataFrame(data_lst2, columns=columns2)
    
    ## Plot the data
    palette ={
            "A100": (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
            "MI50": (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
            "K20": (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
            "TITAN_RTX": (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
            "V100": (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
            "P100": (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
            "GTX_1080Ti": (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
            "RTX_2070_SUPER":(0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
            "GTX_Titan_X": (0.8, 0.7254901960784313, 0.4549019607843137)
            }

    sns.set_theme(style="whitegrid", palette="muted")
    sns.set_context("paper", rc={"font.size":10,"axes.titlesize":7,"axes.labelsize":12})
    sns.set(font_scale = 1.4)

    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(19)

    gpu_order = ["V100", "A100", "P100", "GTX_1080Ti", "GTX_Titan_X", "MI50", "RTX_2070_SUPER", "K20", "TITAN_RTX"]

    g = sns.boxplot(x="GPU", y="Fitness_minima", data=fitdf, palette=palette, order=gpu_order, whis=[0, 100])
    g = sns.stripplot(x="GPU", y="Fitness_minima", data=fitdf, palette=palette, order=gpu_order)
    g.set_title("Fraction of global optimal fitness for minima (convolution)", fontdict={'fontsize': 22})

    g.set_xlabel("GPU", fontsize=20)
    g.set_ylabel("Fraction of global fitnes", fontsize=20)
    plt.show()
