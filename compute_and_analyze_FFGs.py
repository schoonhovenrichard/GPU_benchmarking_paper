import numpy as np
import csv
import json
import os
import itertools
import warnings
import matplotlib.pyplot as plt
import networkx as nx
import pickle

import bloopy
from bloopy.individual import individual, continuous_individual
import bloopy.utils as utils
import bloopy.analysis.analysis_utils as anutil
import bloopy.analysis.critical_points as critpts
import bloopy.analysis.FFG
import gpu_utils

if __name__ == '__main__':
    np.set_printoptions(precision=4)

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

    #NOTE: Choose the GPU-kernel combination you wish to analyze
    for filename in convolution_files[4:5]:
        with open(data_path + filename, 'r') as myfile:
            data=myfile.read()
        data = json.loads(data)

        print("\nDevice: " + str(data['device_name']))
        print("Kernel name: " + str(data['kernel_name']))
        print("Tunable parameters: " + str(data['tune_params_keys']), end='\n\n')

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
        bestkey = None
        for k in data['cache'].keys():
            time = data['cache'][k]['time']
            if time < best_fit:
                best_fit = time
                bestkey = k
        #print("Optimal settings in cache are:", bestkey, "with time {0:.4f}".format(best_fit))
        print("There are", len(data['cache'].keys()), "keys in the searchspace")

        ###  <<<  ANALYZING SEARCH SPACES  >>>
        #method = 'circular'
        method = 'bounded'
        #method = 'Hamming'
        boundary_list = utils.generate_boundary_list(searchspace)
        indiv = individual(bsize, boundary_list=boundary_list)

        ## Find the global minimum
        best_key_bs = gpu_utils.convert_gpusetting_to_bitidxs(bestkey, boundary_list, searchspace_orig)
        utils.set_bitstring(indiv, list(best_key_bs))
        glob_fit = disc_space.fitness(indiv.bitstring)
        print("Global minimum:", bestkey, "with fitness", glob_fit)

        
        ## Loop through the space and assign point types to each point.
        ## Also build the space dictionary.
        nidxs_dict = anutil.build_nodeidxs_dict(boundary_list, disc_space.fitness, bsize)
        tot, minimas, maximas, saddles, regulars, spacedict = critpts.classify_points(bsize, boundary_list, nidxs_dict, method=method)

        idxs_to_pts = anutil.indices_to_points(spacedict)
        print(tot, minimas, maximas, saddles, regulars)

        ###   COMPUTE FFG   ###
        G = bloopy.analysis.FFG.build_FFG(nidxs_dict, boundary_list, method=method)


        ###   ANALYZE FFG   ###
        graph_name = "FFG_data/FFG_" + method + "_" + filename[:-5] + ".txt"
        ## Check graph properties such as cycles
        globopt_idx = spacedict[tuple(best_key_bs)][2]
        print("Global optimum is node:", globopt_idx)
        print(len(G.nodes()), "nodes,", len(G.edges()), "edges, in search space graph")

        ## Calculate centralities
        #centrality = "degree"
        #centrality = "katz"
        centrality = "pagerank"
        #centrality = "closeness"
        if centrality == "degree":
            # Degree centrality can be seen as the immediate risk of a node for catching whatever is flowing through the network.
            # So a high degree centrality for global optimum means there is
            # a better chance of ending up there.
            centrality_dict = nx.algorithms.centrality.degree_centrality(G)
            print("Degree centrality of global optimum:", centrality_dict[globopt_idx])
        elif centrality == "eigen":
            # Eigen vector centraliy is similar, it is a measure of the influence
            # of a node in a network.
            centrality_dict = nx.eigenvector_centrality_numpy(G.reverse())
            print("Eigen vector centrality of global optimum:", centrality_dict[globopt_idx])
        elif centrality == "katz":
            # Katz centrality is a variant of eigenvector, look up details
            centrality_dict = nx.katz_centrality_numpy(G)
            print("Katz vector centrality of global optimum:", centrality_dict[globopt_idx])
        elif centrality == "secondorder":
            # The second order centrality of a given node is the standard
            # deviation of the return times to that node of a perpetual
            # random walk on G.
            centrality_dict = nx.algorithms.centrality.second_order_centrality(G)
            print("Second order centrality of global optimum:", centrality_dict[globopt_idx])
        elif centrality == "closeness":
            centrality_dict = nx.algorithms.centrality.closeness_centrality(G)
            print("Closeness centrality of global optimum:", centrality_dict[globopt_idx])
        elif centrality == "pagerank":
            centrality_dict = nx.algorithms.link_analysis.pagerank_alg.pagerank(G)
            print("Pagerank centrality of global optimum:", centrality_dict[globopt_idx])
        else:
            raise Exception("Unknown centrality type")

        centr_name = "FFG_data/propFFG_centrality_" + centrality + "_" + method + "_" + filename[:-5] + ".csv"
        percs = np.arange(0.0, 0.16, 0.01).tolist()
        centralities = [["Percentage","proportion_centr","sum_accept_centr", "tot_centr", "minima_centr", "nr_of_nodes"]]
        for perc in percs:
            acceptable_minima = critpts.strong_local_minima(perc, glob_fit, spacedict)
            accept_centr, tot_centr, minima_centr = bloopy.analysis.FFG.average_centrality_nodes(centrality_dict, acceptable_minima, spacedict, idxs_to_pts)
            prop_centr = accept_centr/float(minima_centr)
            centralities.append([perc, prop_centr, accept_centr, tot_centr, minima_centr, len(centrality_dict.values())])
            print("Proportion of centrality of strong local minima", perc, ":",prop_centr)
        # Save to CSV file
        with open(centr_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(centralities)

        #NOTE: Uncomment continue if you only want to compute pagerank centralities
        continue

        ## Plot the graph with NetworkX
        color_map = []
        size_map = []
        threshold = 1.25
        cmax = threshold * glob_fit
        for node in G:
            pt = idxs_to_pts[node][0]
            if spacedict[pt][0] == 1:
                fit = spacedict[pt][1]
                color_map.append(fit)
                if fit > cmax:#70 for convolution
                    siz = 3 + 30*(glob_fit/float(cmax) - 1/float(threshold))
                else:
                    siz = 3 + 30*(glob_fit/float(fit) - 1/float(threshold))
                size_map.append(siz)
            else:
                fit = spacedict[pt][1]
                color_map.append(fit)
                size_map.append(0.28)

        if nx.is_directed(G):
            arz = 2
            arst = '-|>'
            wdth = 0.07
            #pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
            #nx.draw(G, pos=pos, node_color=color_map, arrowsize=arz, node_size=size_map, with_labels=False, arrows=True,arrowstyle=arst, font_size=1, cmap=plt.get_cmap('viridis_r'), vmin=glob_fit, vmax=cmax, width=wdth)

            nx.draw_kamada_kawai(G, arrows=True, arrowstyle=arst, arrowsize=arz, node_size=size_map, node_color=color_map, font_size=1, with_labels=False, cmap=plt.get_cmap('viridis_r'), vmin=glob_fit, vmax=cmax, width=wdth)


            #nx.draw(G, arrows=True, arrowstyle=arst, arrowsize=arz, node_size=size_map, node_color=color_map, font_size=1, with_labels=False, cmap=plt.get_cmap('viridis_r'), vmin=glob_fit, vmax=cmax, width=wdth)
            #nx.draw_circular(G, arrows=True, arrowstyle=arst, arrowsize=arz, node_size=size_map, node_color=color_map, font_size=1, with_labels=True, cmap=plt.get_cmap('viridis_r'), vmin=glob_fit, vmax=cmax, width=wdth)
            #nx.draw_shell(G, arrows=True, arrowstyle=arst, arrowsize=arz, node_size=size_map, node_color=color_map, font_size=1, with_labels=True, cmap=plt.get_cmap('viridis_r'), vmin=glob_fit, vmax=cmax, width=wdth)
            #nx.draw_spring(G, arrows=True, arrowstyle=arst, arrowsize=arz, node_size=size_map, node_color=color_map, font_size=1, with_labels=True, cmap=plt.get_cmap('viridis_r'), vmin=glob_fit, vmax=cmax, width=wdth)
        else:
            nx.draw_kamada_kawai(G, node_size=size_map, node_color=color_map, font_size=1, with_labels=True, cmap=plt.get_cmap('viridis_r'), vmin=glob_fit, vmax=cmax, width=0.35)
            #nx.draw(G, node_size=3, node_color=color_map, font_size=1, with_labels=True, cmap=plt.get_cmap('viridis_r'), vmin=glob_fit, vmax=1.15*glob_fit)
            #nx.draw_circular(G, node_size=3, node_color=color_map, font_size=1, with_labels=True, cmap=plt.get_cmap('viridis_r'), vmin=glob_fit, vmax=1.15*glob_fit)
            #nx.draw_shell(G, node_size=3, node_color=color_map, font_size=1, with_labels=True, cmap=plt.get_cmap('viridis_r'), vmin=glob_fit, vmax=1.15*glob_fit)
            #nx.draw_spring(G, node_size=3, node_color=color_map, font_size=1, with_labels=True, cmap=plt.get_cmap('viridis_r'), vmin=glob_fit, vmax=1.15*glob_fit)

        plt.axis('off')
        plt.draw()
        plt.savefig(graph_name[:-4] + ".pdf")
        plt.clf()
        del G
        del spacedict
        del centrality_dict
        del acceptable_minima
        del idxs_to_pts
