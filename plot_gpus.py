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
    experiment_dir = root_dir + 'GPU_benchmarking_paper/experimental_data/'

    algos = [
        "RandomSampling",
        "BasinHopping",
        "SimulatedAnnealing",
        "GeneticAlgorithm",
        "GLS",
        "RandomGreedyTabu",
        "BestTabu",
        "RandomGreedyILS",
        "BestILS",
        "RandomGreedyMLS",
        "BestMLS",
        "ParticleSwarm",
        "DifferentialEvolution",
        "DualAnnealing",
        ]
    maxfevals = [25,50,75,100,150,200,400,600,800,1000,2000,1000000]

    kernel = "convolution"
    #kernel = "GEMM"
    #kernel = "pnpoly"
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

    ### Read the files
    # The files are structured as:
    #  'Algorithm', 'Mean fraction of optimum', 'StDev fraction of optimum', 'Success rate', 'Mean function evaluations', 'StDev function evaluations', 'Settings', 'MaxFEval'
    data_algos = dict()
    for alg in algos:
        algdata = []
        for f in data_files[alg]:
            with open(file_dir + f, 'r') as read_obj:
                csv_reader = csv.reader(read_obj)
                list_data = list(csv_reader)[2:]
                algdata.append(list_data)
        data_algos[alg] = dict()
        alg_list = []
        for i in range(len(algdata)):
            for j in range(len(algdata[i])):
                alg_list.append(algdata[i][j][1:-1])
        data_algos[alg] = alg_list

    # Now we have aggregated all data per algorithm.
    # We must place each data point in the correct bin
    binned_data = dict()
    for alg in algos:
        algdat = data_algos[alg]
        bindict = dict()
        for mfev in maxfevals:
            bindict[mfev] = []
        for point in algdat:
            fevmean = float(point[3])
            for mfev in maxfevals:
                if fevmean <= mfev:
                    bindict[mfev] = bindict[mfev] + [point]
                    break
        binned_data[alg] = bindict

    ### Create dictionary to contain plot data
    plot_dict = dict()
    for alg in algos:
        lst = []
        for mfev in maxfevals:
            #plotdat = [mfev]
            plotdat = []
            if len(binned_data[alg][mfev]) > 0:
            #if len(data_algos[alg][mfev]) > 0:
                #dat = data_algos[alg][mfev]
                dat = binned_data[alg][mfev]
                # To combine two means and Stdevs of different distributions,
                # There are closed form expressions. However, samples n_i per GPU
                # are equal here, significantly simplifying the result.
                fracmean = 0.0
                fracstd = 0.0
                success = 0.0
                fevmean = 0.0
                fevstd = 0.0
                y = 1/float(len(dat))
                for dtp in dat:
                    fracmean += y * float(dtp[0])
                    success += y * float(dtp[2])
                    fevmean += y * float(dtp[3])
                    # add as variances first
                    fracstd += y * float(dtp[1])**2
                    fevstd += y * float(dtp[4])**2
                # Turn variances into stdev
                fracstd = np.sqrt(fracstd)
                fevstd = np.sqrt(fevstd)
                plotdat.append([fracmean, fracstd])
                plotdat.append(success)
                plotdat.append([fevmean, fevstd])
                lst.append(plotdat)
        plot_dict[alg] = lst


    ### NEW SEABORN STUFF
    GPUs = ["A100", "RTX_2070_SUPER", "TITAN_RTX", "MI50", "V100", "K20", "GTX_Titan_X", "GTX_1080Ti","P100"]
    data_algos = dict()
    dataframe_lst = []
    for alg in algos:
        algdata = []
        for f in data_files[alg]:
            gpuname = None
            for gp in GPUs:
                if gp in f:
                    gpuname = gp
            if gpuname is None:
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

    ## DEFINE COLOUR PALETTE:
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
    markers = {"A100": "o",
            "MI50": "<",
            "GTX_Titan_X": ".",
            "K20": "v",
            "V100": "s",
            "P100": ">",
            "GTX_1080Ti": "P",
            "RTX_2070_SUPER": "X",
            "TITAN_RTX": "D"
            }
    linestyles = {
            'K20': (4, 1.5),
            'P100': (1, 1),
            'GTX_Titan_X': (3, 1.25, 1.5, 1.25),
            'TITAN_RTX': (5, 1, 1, 1),
            'A100': "",
            #'A100': (3, 1.25, 1.25, 1.25, 1.25, 1.25),
            'V100': (4, 1, 4, 1, 1, 1),
            'MI50': (3, 1.25, 3, 1.25, 1.25, 1.25),
            'RTX_2070_SUPER': "",
            #'RTX_2070_SUPER': (4, 1, 1, 1, 1, 1),
            'GTX_1080Ti': (3, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25)
            }

    ### Make the plots
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

    # SEABORN
    # Select subset to plot
    #NOTE: Choose subset of algorithm you wish to plot
    #subset = ["GreedyILS"]
    subset = ["DualAnnealing"]
    frames = []
    for sub in subset:
        frames.append(fulldf[fulldf.Algorithm == sub])
    plotdf = pd.concat(frames)

    # Create the SEABORN plot
    sns.set_theme(style="whitegrid", palette="muted")
    sns.set_context("paper", rc={"font.size":10,"axes.titlesize":7,"axes.labelsize":12})
    sns.set(font_scale = 1.35)

    fig, ax = plt.subplots()
    fig.set_figheight(7)
    fig.set_figwidth(11)
    g = sns.lineplot(data=plotdf, y='Fraction_optim', x='Func_evals', hue='GPU', style="GPU", markers=markers, linewidth=2.5, ax=ax, dashes=linestyles, palette=palette)
    g.set_title("{0} performance for {1} per GPU".format(subset[0], kernel), fontdict={'fontsize': 26})
    g.set_xlabel("Average function evaluations used", fontsize=22)
    g.set_ylabel("Fraction of optimum", fontsize=22)

    legend_properties = {'size':20}
    legendMain=g.legend(prop=legend_properties)

    #ax.set(xscale="log")
    plt.show()
