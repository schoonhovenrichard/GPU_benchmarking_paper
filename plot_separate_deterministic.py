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
    graphs = False
    grid_graphs = True

    subset1 = ["GreedyILS","GreedyMLS","DualAnnealing", "SimulatedAnnealing", "GLS"]
    subset2 = ["GeneticAlgorithm", "BestMLS", "BestILS", "BasinHopping", "DifferentialEvolution"]
    subset3 = ["SMAC4BB", "GreedyTabu", "BestTabu", "ParticleSwarm","RandomSampling"]

    if graphs:
        testGPUs = ["A100", "TITAN_RTX", "MI50", "V100", "K20", "GTX_Titan_X"]
        #for i in range(len(testGPUs)):
        frames = []
        for sub in subset1:
            frames.append(fulldf[fulldf.Algorithm == sub])
        plotdf1 = pd.concat(frames)
        frames = []
        for gpu in testGPUs[0:1]:
            frames.append(plotdf1[plotdf1.GPU == gpu])
        plotdf = pd.concat(frames)


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
        # Create the SEABORN plot
        sns.set_theme(style="whitegrid", palette="muted")
        sns.set_context("paper", rc={"font.size":10,"axes.titlesize":7,"axes.labelsize":12})
        sns.set(font_scale = 1.35)

        # Select subset to plot
        #g = sns.jointplot(data=fulldf, y='Fraction_optim', x='Func_evals', hue='Algorithm', ylim=(0.3, 1.05), xlim=(-200,6600), space=0.05, height=6.5)
        #g.set_axis_labels("Average function evaluations used", "Fraction of optimum", size=15)
        #legend_properties = {'size':18}
        #legend_properties = {'weight':'bold','size':18}
        #legendMain=g.ax_joint.legend(prop=legend_properties,loc='lower right')
        #g.fig.suptitle('Fraction of optimal GPU setting')

        fig, ax = plt.subplots()
        fig.set_figheight(7)
        fig.set_figwidth(11)
        g = sns.lineplot(data=plotdf, x="Func_evals", y='Fraction_optim', hue='Algorithm', ax=ax)
        g.set(xscale="log")
        g.set(xlim=(25, 6400))
        ticks = budgets
        g.set(xticks=ticks)
        g.set(xticklabels=ticks)

        g.set_title('Fraction of optimal fitness (stochastic) for {0}'.format(kernel), size=24)
        g.set_xlabel("Max budget", fontsize=21)
        g.set_ylabel("Fraction of optimal fitness", fontsize=21)
        ax.tick_params(labelsize=13)

        legend_properties = {'size':20}
        legendMain=g.legend(prop=legend_properties, loc='lower right')

        #fig,ax = plt.subplots()
        #for gpu_model, subdf in fulldf.groupby('GPU'):
        #    ax = sns.lineplot(data=subdf, y='Fraction_optim', x='Func_evals', hue='Algorithm')

        #ax = sns.jointplot(data=fulldf, y='Fraction_optim', x='Func_evals', hue='Algorithm', kind="kde", levels=5)
        #ax.plot_joint(sns.kdeplot, zorder=0, levels=5)
        #ax = sns.boxplot(data=fulldf, x="GPU", y="Fraction_optim", hue="Algorithm", order=GPUs)
        plt.show()


    if grid_graphs:
        plotGPUs = ["A100", "TITAN_RTX", "MI50", "V100", "K20", "GTX_Titan_X"]
        frames = []
        thesubset = subset3
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
