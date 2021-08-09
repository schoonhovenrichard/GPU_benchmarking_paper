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
    pth = 'GPU_benchmarking_paper/FFG_data/'
    experiment_dir = root_dir + pth + 'bounded_pagerank_centrality/'
    
    kernel = "convolution"
    #kernel = "GEMM"
    #kernel = "pnpoly"
    if kernel == "pnpoly":
        GPUs = ["A100", "V100", "RTX_2070_SUPER", "TITAN_RTX", "P100", "K20", "GTX_Titan_X", "GTX_1080Ti"]
    else:
        GPUs = ["A100", "V100", "RTX_2070_SUPER", "TITAN_RTX", "P100", "K20", "GTX_Titan_X", "GTX_1080Ti", "MI50"]

    file_dir = experiment_dir
    exper_files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
    columns = ["GPU", "Percentage", "Prop_centrality", "Sum_accept_centr", "Tot_centr", "Minima_centr", "Tot_nodes"]
    dataframe_lst = []
    for f in exper_files:
        raw = f.split("_")
        if "centrality" not in raw:
            continue
        if kernel not in raw:
            continue

        # Find GPU
        gpu = None
        if "MI50" in raw:
            gpu = "MI50"
        else:
            gpu = "_".join(raw[5:-1])
        if gpu is None:
            print(f.split("_"))
            raise Exception("Something wrong")
        print(gpu)

        # Open the file
        with open(file_dir + f, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            list_data = list(csv_reader)[1:]
        for dat in list_data:
            perc = float(dat[0])
            propcentr = float(dat[1])
            sumacceptcentr = float(dat[2])
            totcentr = float(dat[3])
            minimacentr = float(dat[4])
            totnodes = int(dat[5])
            entry = [gpu, perc, propcentr, sumacceptcentr, totcentr, minimacentr, totnodes]
            dataframe_lst.append(entry)

    plotdf = pd.DataFrame(dataframe_lst, columns=columns)

    ### Make the plots
    # Plot settings
    fm = ''
    cps = 2.0
    linesty = '-'
    import matplotlib
    font = {'family' : 'sans-serif',
#            'weight' : 'bold',
            'size'   : 34}
    matplotlib.rc('font', **font)

    # Create the SEABORN plot
    sns.set_theme(style="whitegrid", palette="muted")
    sns.set_context("paper", rc={"font.size":10,"axes.titlesize":7,"axes.labelsize":12})
    sns.set(font_scale = 1.35)

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
            'V100': (4, 1, 4, 1, 1, 1),
            'MI50': (3, 1.25, 3, 1.25, 1.25, 1.25),
            'RTX_2070_SUPER': "",
            'GTX_1080Ti': (3, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25)
            }

#    #style palette
    fig, ax = plt.subplots()
    fig.set_figheight(7)
    fig.set_figwidth(11)
    g = sns.lineplot(data=plotdf, y='Prop_centrality', x='Percentage', hue='GPU', style='GPU', linewidth=2.5, ax=ax, palette=palette, markers=markers, dashes=linestyles)
    g.set_title("Proportion of centrality for {0} per GPU".format(kernel), fontdict={'fontsize': 26})
    g.set_xlabel("Percentage acceptable minima", fontsize=22)
    g.set_ylabel("Proportion of centrality", fontsize=22)

    legend_properties = {'size':14}
    legendMain=g.legend(prop=legend_properties)
    
    ax.set(yscale="log")
    #ax.set(ylim=(0.012627420042927072, 1.0))#GEMM
    ax.set(ylim=(0.004311175781390637, 1.0))#Convolution
    plt.show()
