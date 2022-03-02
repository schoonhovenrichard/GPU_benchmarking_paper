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
    experiment_dir = root_dir + 'GPU_benchmarking_paper/experimental_data/stochastic/'

    algos = [
        "RandomGreedyILS",
        "DualAnnealing",
        "GeneticAlgorithm",
        "SMAC4BB",
        'irace',
        ]
    budgets = [25,50,100,200,400,800,1600,3200,6400]

    ###    LOAD DATA    ###
    kernel = "convolution"
    #kernel = "GEMM"
    #kernel = "pnpoly"

    #fevalrange = 'lowrange'
    #fevalrange = 'highrange'
    fevalrange = 'all'


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
    plot_lst = []
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
        columns = ["Algorithm", "GPU", "Func_evals", "Func_evals_std", "Fraction_optim", "Fraction_optim_std", "Success_rate", "Max_budget"]
        for gpudat in algdata:
            entries = [[] for i in range(len(budgets))]
            for dat in gpudat:
                fracopt = float(dat[1])
                fracopt_std = float(dat[2])
                success = float(dat[3])
                feval = float(dat[4])
                feval_std = float(dat[5])
                gpu = dat[-1]
                maxfev = int(dat[-2])
                if "RandomGreedy" in alg:
                    alg = alg[6:]

                for i, budget in enumerate(budgets):
                    if feval <= 1.03*budget:
                        entries[i] = [alg, gpu, feval, feval_std, fracopt, fracopt_std, success, maxfev]
                        break
                entry = [alg, gpu, feval, feval_std, fracopt, fracopt_std, success, maxfev]
                plot_lst.append(entry)
            for entry in entries:
                if len(entry) == 0:
                    dataframe_lst.append([alg, gpu, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
                else:
                    dataframe_lst.append(entry)
    fulldf = pd.DataFrame(dataframe_lst, columns=columns)
    plotdf = pd.DataFrame(plot_lst, columns=columns)

    ### PLOT TYPE
    graphs = True
    grid_graphs = False
    competition_grid = False

    if competition_grid:
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

        arr = [[0 for i in range(len(algos))] for alg in algos]
        rowidx = 0
        dflist= []
        GPUs = ["A100", "TITAN_RTX", "MI50", "V100", "K20", "GTX_Titan_X"]# Redefine without P100
        pnpolyGPUs = ["A100", "TITAN_RTX", "V100", "K20", "GTX_Titan_X"]
        comps_performed = 0
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
                        b = 6# <=800
                    elif fevalrange == 'highrange':
                        a = 6
                        b = min(l2.shape[0], comparel2.shape[0])
                    elif fevalrange == 'all':
                        a = 0
                        b = min(l2.shape[0], comparel2.shape[0])
                    for x in range(a ,b):
                        if l2.iloc[x][4] == -np.inf or comparel2.iloc[x][4] == -np.inf:
                            continue # Do not perform competition if one has no data on this
                        fo = l2.iloc[x].iloc[4]
                        fostd = l2.iloc[x].iloc[5]
                        compfo = comparel2.iloc[x].iloc[4]
                        compfostd = comparel2.iloc[x].iloc[5]
                        ite += 1
                        if fo > compfo:
                            comps_performed += 1
                            stat = ttest_ind_from_stats(fo, fostd, N, compfo, compfostd, N)
                            if stat.pvalue < alpha:
                                count += 1
                                algcount += 1
                entry = [compare, algo, algcount]
                dflist.append(entry)
                arr[rowidx][colidx] = algcount
                colidx += 1
            rowidx += 1
            print("Total number of statistical tests performed:", comps_performed)
            print(compare, "In total,", count, "times another algorithm was better out of", ite, "times")

        competitiondf = pd.DataFrame(dflist, columns=["Algorithm1", "Algorithm2", "1beats2"])


        ordered_algos = [
            "DualAnnealing",
            "RandomGreedyILS",
            "GeneticAlgorithm",
            "SMAC4BB",
            'irace',
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

    if graphs:
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

        optimum_frac = True
        success_rate = False

        # SEABORN
        # Create the SEABORN plot
        sns.set_theme(style="whitegrid", palette="muted")
        sns.set_context("paper", rc={"font.size":10,"axes.titlesize":7,"axes.labelsize":12})
        sns.set(font_scale = 1.35)

        # Select subset to plot
        #g = sns.jointplot(data=plotdf, y='Fraction_optim', x='Func_evals', hue='Algorithm', ylim=(0.3, 1.05), xlim=(-200,6600), space=0.05, height=6.5)
        #g.set_axis_labels("Average function evaluations used", "Fraction of optimum", size=15)
        #legend_properties = {'size':18}
        #legend_properties = {'weight':'bold','size':18}
        #legendMain=g.ax_joint.legend(prop=legend_properties,loc='lower right')
        #g.fig.suptitle('Fraction of optimal GPU setting')

        fig, ax = plt.subplots()
        fig.set_figheight(7)
        fig.set_figwidth(11)
        g = sns.lineplot(data=plotdf, x="Max_budget", y='Fraction_optim', hue='Algorithm', ax=ax)
        g.set(xscale="log")
        g.set(xlim=(25, 6400))
        ticks = budgets
        g.set(xticks=ticks)
        g.set(xticklabels=ticks)

        g.set_title('Fraction of optimal fitness (stochastic) for {0}'.format(kernel), size=19)
        g.set_xlabel("Max budget", fontsize=21)
        g.set_ylabel("Fraction of optimal fitness", fontsize=21)
        ax.tick_params(labelsize=13)

        legend_properties = {'size':20}
        legendMain=g.legend(prop=legend_properties, loc='lower right')

        #fig,ax = plt.subplots()
        #for gpu_model, subdf in plotdf.groupby('GPU'):
        #    ax = sns.lineplot(data=subdf, y='Fraction_optim', x='Func_evals', hue='Algorithm')

        #ax = sns.jointplot(data=plotdf, y='Fraction_optim', x='Func_evals', hue='Algorithm', kind="kde", levels=5)
        #ax.plot_joint(sns.kdeplot, zorder=0, levels=5)
        #ax = sns.boxplot(data=plotdf, x="GPU", y="Fraction_optim", hue="Algorithm", order=GPUs)
        plt.show()


    if grid_graphs:
        plotGPUs = ["A100", "TITAN_RTX", "MI50", "V100", "K20", "GTX_Titan_X"]

        height = 4
        fig, axes = plt.subplots(
            nrows=1,
            ncols=6,
            figsize=(height * 6, height)
        )
        print(axes)
        fig.patch.set_alpha(1.0)
        for ax, gpu in zip(axes, plotGPUs):
            print(ax)
            print(gpu)
            gpudf = plotdf[plotdf.GPU == gpu]
            #sns.lineplot(data=gpudf, y='Fraction_optim', x='Func_evals', hue='Algorithm', ax=ax)
            sns.lineplot(data=gpudf, y='Fraction_optim', x='Max_budget', hue='Algorithm', ax=ax)
            #pcm = ax.imshow(v.squeeze(), cmap=cmap, clim=clim)
            #fig.colorbar(pcm, ax=ax)
            ax.set_title(gpu)
        fig.tight_layout()
        plt.show()
