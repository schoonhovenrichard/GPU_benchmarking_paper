import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import seaborn as sns
import pandas as pd

if __name__ == '__main__':
    ### Get the files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-1]) + "/"
    experiment_dir = root_dir + 'GPU_benchmarking_paper/experimental_data/deterministic/'

    algos = [
        "RandomGreedyMLS",
        "BestMLS",
        "RandomGreedyILS",
        "BestILS",
        "DualAnnealing",
        "GLS",
        "DifferentialEvolution",
        "GeneticAlgorithm",
        "BasinHopping",
        "ParticleSwarm",
        "RandomGreedyTabu",
        "BestTabu",
        "SimulatedAnnealing",
        "RandomSampling",
        "SMAC4BB",
        ]

    budgets = [25,50,100,200,400,800,1600,3200,6400]

    ###    LOAD DATA    ###
    #kernel = "convolution"
    #kernel = "GEMM"
    kernel = "pnpoly"

    file_dir = experiment_dir + kernel + '/'
    exper_files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
    data_files = dict()
    for alg in algos:
        data_files[alg] = []

    for f in exper_files:
        alg = f.split("_")[0]
        if alg in data_files:
            data_files[alg] = data_files[alg] + [f]
        else:
            print(f.split("_"))
            raise Exception("Unknown algorithm found in data file")

    ###    LOAD RELEVANT DATA IN PANDAS DF    ###
    GPUs = ["A100", "RTX_2070_SUPER", "TITAN_RTX", "MI50", "V100", "K20", "GTX_Titan_X", "GTX_1080Ti", "P100"]
    pnpolyGPUs = ["A100", "RTX_2070_SUPER", "TITAN_RTX", "V100", "K20", "GTX_Titan_X", "GTX_1080Ti", "P100"]
    data_algos = dict()
    dataframe_lst = []
    for alg in algos:
        algdata = []
        for f in data_files[alg]:
            gpuname = None
            for gp in GPUs:
                if gp in f:
                    gpuname = gp
            if gpuname == "P100" or gpuname == 'GTX_1080Ti' or gpuname == 'RTX_2070_SUPER':#WAS USED FOR TUNING, NOT COUNTED HERE
                continue
            elif gpuname is None:
                print(f)
                raise Exception("Unknown GPU in files?")
            with open(file_dir + f, 'r') as read_obj:
                csv_reader = csv.reader(read_obj)
                list_data = list(csv_reader)[2:]
                for k in range(len(list_data)):
                    list_data[k].append(gpuname)
                algdata.append(list_data)
        columns = ["Algorithm", "GPU", "Func_evals", "Func_evals_std", "Fraction_optim", "Fraction_optim_std", "Success_rate"]
        for gpudat in algdata:
            for dat in gpudat:
                fracopt = float(dat[1])
                fracopt_std = float(dat[2])
                success = float(dat[3])
                feval = float(dat[4])
                feval_std = float(dat[5])
                gpu = dat[-1]
                if "RandomGreedy" in alg:
                    alg = alg[6:]
                entry = [alg, gpu, feval, feval_std, fracopt, fracopt_std, success]
                dataframe_lst.append(entry)
    fulldf = pd.DataFrame(dataframe_lst, columns=columns)


    ### PLOT TYPE
    subset1 = ["GreedyILS","GreedyMLS","DualAnnealing", "SimulatedAnnealing", "GLS"]
    subset2 = ["GeneticAlgorithm", "BestMLS", "BestILS", "BasinHopping", "DifferentialEvolution"]
    subset3 = ["SMAC4BB", "GreedyTabu", "BestTabu", "ParticleSwarm","RandomSampling"]
    plotGPUs = ["A100", "TITAN_RTX", "MI50", "V100", "K20", "GTX_Titan_X"]

    # NOTE: WHICH ALGORITHMS TO PLOT
    thesubset = subset3

    frames = []
    for sub in thesubset:
        frames.append(fulldf[fulldf.Algorithm == sub])
    plotdf = pd.concat(frames)

    height = 12
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(height * 1.75, height)
    )
    print(axes)
    fig.patch.set_alpha(1.0)
    for ax, gpu in zip(axes.flatten(), plotGPUs):
        print(ax)
        print(gpu)
        gpudf = plotdf[plotdf.GPU == gpu]
        #sns.lineplot(data=gpudf, y='Fraction_optim', x='Func_evals', hue='Algorithm', ax=ax)
        #pcm = ax.imshow(v.squeeze(), cmap=cmap, clim=clim)
        #fig.colorbar(pcm, ax=ax)

        ps = []
        for algo in thesubset:
            print(algo)
            algodf = gpudf[gpudf.Algorithm == algo]
            algodf = algodf.sort_values(by=['Func_evals'], ascending=True)
            ps.append(ax.errorbar(algodf['Func_evals'], algodf['Fraction_optim'], yerr=algodf['Fraction_optim_std'], label=algo, capsize=2,fmt=''))
            #ps.append(ax.errorbar(algodf['Fraction_optim'], algodf['Func_evals'], yerr=algodf['Fraction_optim_std'], label=algo, color='g', ecolor='g', capsize=2,fmt=''))
            ax.set_xscale('log')
            ax.set_title(gpu, fontsize=18)
            ax.set_xlabel("Max budget", fontsize=15)
            ax.set_ylabel("Fraction of optimal fitness", fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=8)


    plt.legend(handles=ps, loc='lower right', prop={'size': 19})
    fig.suptitle("Algorithm fraction of optimum per GPU for "+kernel, fontsize=26)
    fig.tight_layout()
    plt.show()
