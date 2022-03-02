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

    ###    LOAD DATA    ###
    kernel = "convolution"
    #kernel = "GEMM"
    #kernel = "pnpoly"

    fevalrange = 'lowrange'
    #fevalrange = 'highrange'


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

    ###    PERFORM COMPETITIONS    ###
    from scipy.stats import ttest_ind_from_stats
    N = 100
    alpha = 0.05

    # Plot settings
    fm = ''
    cps = 2.0
    #linesty = 'None'
    linesty = '-'
    import matplotlib
    font = {'family' : 'sans-serif',
#            'weight' : 'bold',
            'size'   : 34}

    matplotlib.rc('font', **font)

    arr = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for alg in algos]
    rowidx = 0
    dflist= []
    GPUs = ["A100", "TITAN_RTX", "MI50", "V100", "K20", "GTX_Titan_X"]# Redefine without P100
    pnpolyGPUs = ["A100", "TITAN_RTX", "V100", "K20", "GTX_Titan_X"]
    for compare in algos:
        if "RandomGreedy" in compare:
            compare = compare[6:]
        count = 0
        ite = 0
        comparel1 = fulldf[fulldf.Algorithm == compare]
        colidx = 0
        for algo in algos:
            if "RandomGreedy" in algo:
                algo = algo[6:]
            l1 = fulldf[fulldf.Algorithm == algo]
            algcount = 0
            if kernel == 'pnpoly':
                theGPUs = pnpolyGPUs
            else:
                theGPUs = GPUs
            for gpu in theGPUs:
                l2 = l1[l1.GPU == gpu]
                if l2.size == 0:
                    continue

                comparel2 = comparel1[comparel1.GPU == gpu]
                if comparel2.size == 0:
                    continue
                if fevalrange == 'lowrange':
                    a = 0
                    b = 4
                elif fevalrange == 'highrange':
                    a = 4
                    b = min(l2.shape[0], comparel2.shape[0])
                for x in range(a ,b):
                    fo = l2.iloc[x].iloc[4]
                    fostd = l2.iloc[x].iloc[5]
                    test = comparel2.iloc[x]
                    compfo = comparel2.iloc[x].iloc[4]
                    compfostd = comparel2.iloc[x].iloc[5]
                    ite += 1
                    if fo > compfo:
                        stat = ttest_ind_from_stats(fo, fostd, N, compfo, compfostd, N)
                        if stat.pvalue < alpha:
                            count += 1
                            algcount += 1
            entry = [compare, algo, algcount]
            dflist.append(entry)
            arr[rowidx][colidx] = algcount
            colidx += 1
        rowidx += 1
        print(compare, "In total,", count, "times another algorithm was better out of", ite, "times")

    competitiondf = pd.DataFrame(dflist, columns=["Algorithm1", "Algorithm2", "1beats2"])


    ordered_algos = [
        "BasinHopping",
        "DualAnnealing",
        "DifferentialEvolution",
        "ParticleSwarm",
        "RandomGreedyILS",
        "BestILS",
        "RandomGreedyTabu",
        "BestTabu",
        "RandomGreedyMLS",
        "BestMLS",
        "SimulatedAnnealing",
        "GLS",
        "GeneticAlgorithm",
        "SMAC4BB",
        "RandomSampling",
        ]

    count = 0
    print('\nWINS:')
    for algo in ordered_algos:
        if "RandomGreedy" in algo:
            algo = algo[6:]
        print(algo+':', competitiondf[competitiondf.Algorithm2 == algo]['1beats2'].sum())
        count += competitiondf[competitiondf.Algorithm2 == algo]['1beats2'].sum()
    print("COUNT:", count)

    count = 0
    print('\nLOSSES:')
    for algo in ordered_algos:
        if "RandomGreedy" in algo:
            algo = algo[6:]
        print(algo+':', competitiondf[competitiondf.Algorithm1 == algo]['1beats2'].sum())
        count += competitiondf[competitiondf.Algorithm1 == algo]['1beats2'].sum()
    print("COUNT:", count)


    ### Make the seaborn heatmap
    sns.set_context("paper", rc={"font.size":1,"axes.titlesize":1,"axes.labelsize":1})
    sns.set(font_scale = 0.7)
    colormap = sns.color_palette("coolwarm", as_cmap=True)

    plotdf = competitiondf.pivot("Algorithm1", "Algorithm2", "1beats2")

    fig, ax = plt.subplots(1, 1, figsize = (15, 15), dpi=300)

    g = sns.heatmap(plotdf, annot=True, fmt="d", linewidths=.5, cmap=colormap, cbar=False)

    ax.set_ylabel('')
    ax.set_xlabel('')
    #NOTE: Set correct title
    title_str = "Algorithm Column beats Row - " + kernel
    if fevalrange == 'lowrange':
        title_str += ' feval <= 200'
    elif fevalrange == 'highrange':
        title_str += ' feval > 200'

    ax.set_title(title_str, fontsize=9)

    fig.set_size_inches(5, 4.5)
    plt.show()
