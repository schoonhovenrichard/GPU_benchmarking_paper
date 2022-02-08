import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import warnings

def find_data_file(algo, kernelname, experiment_dir):
    exper_files = [f for f in os.listdir(experiment_dir) if os.path.isfile(os.path.join(experiment_dir, f))]
    data_files = []
    for f in exper_files:
        sls = f.split("_")
        if len(sls) < 4:
            continue
        elif sls[2] == algo:
            if sls[3] == kernelname:
                data_files.append(f)
    if len(data_files) == 0:
        raise Exception("Could not find file for this algorithm.")
    return data_files

def read_process_file(experiment_dir, data_file):
    ### Read the file
    with open(experiment_dir + data_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        list_data = list(csv_reader)

    hyperdata = []
    for i in range(2, len(list_data)):
        dat = [list_data[i][0],
                float(list_data[i][1]),
                float(list_data[i][2]),
                float(list_data[i][3]),
                float(list_data[i][4]),
                float(list_data[i][5])]
        dat += list_data[i][6:]
        hyperdata.append(dat)
    hyperdata.sort(key=lambda x: x[4]) # Sort on function evaluations
    return hyperdata

def partition_feval(hyperdata, maxfevals):
    # Partition in right max_feval category
    #maxfevals = [50,100,150,200,400,600,800,1000,2000,1000000]#One extra for overflow
    maxfevals2 = maxfevals + [1000000]#One extra for overflow
    fevalpartition = [[] for x in range(len(maxfevals2))]
    idx = 0
    error = 0.05
    found_entries = []# Some algorithms never use many fevals, check if bin is relevant
    for i in range(len(maxfevals2)):
        found = False
        for xs in hyperdata:
            if xs[4] <= (1+error)*maxfevals2[i]:
                fevalpartition[i].append(xs)
                if i > 0:
                    if xs[4] > (1+error)*maxfevals2[i-1]:
                        found = True
                else:
                    found = True
        if not found:
            print("No elements in this partition:", maxfevals2[i])
        found_entries.append(found)
    for x in fevalpartition:
        x.sort(key=lambda x: -x[1])
    return fevalpartition, found_entries

def clean_up_hyperpars(lst):
    if len(lst) > 1:
        #raise Exception("UNEXPECTED")
        warnings.warn("Unexpected already split")
        splitted = lst
    elif len(lst) == 1:
        splitted = lst[0].split(';')
    else:
        raise Exception("Empty list argument lst")
    new = []
    for hyperpar in splitted:
        key, val = hyperpar.split("=")
        if " at " in val:
            nval = (val.split(" at ")[0]).split(" ")[-1]
            val = nval
        new.append(key+"="+val+";")
    returnstr = [''.join(new)[:-1]]
    return tuple(returnstr)

def get_best_settings(fevalpartition, margin, maxfevals):
    # Select best settings
    #maxfevals = [50,100,150,200,400,600,800,1000,2000,1000000]#One extra for overflow
    maxfevals2 = maxfevals + [1000000]#One extra for overflow
    fevalsettings = [[] for x in range(len(maxfevals2)-1)]
    for i in range(len(maxfevals2)-1):
        if len(fevalpartition[i]) == 0:
            continue
        best = fevalpartition[i][0]
        print(best[1], best[2])
        for x in range(len(fevalpartition[i])):
            elem = fevalpartition[i][x]
            #if elem[1] <  best[1] - margin * best[2]:
            if elem[1] <  best[1] - margin:
                break
            entry = clean_up_hyperpars(elem[6:])
            #NOTE: For differential evolution, we have to remove iterations as it depends
            # only on pop_size and problem bitstring size.
            devoent = entry[0].split(";")[:-1]
            devoentry = tuple(["".join(devoent)])
            entry = devoentry
            fevalsettings[i].append(entry)#Need hashable for sets later
    return fevalsettings

def homogenize_dict(z):
    #If there are many settings, try to find those that are similar across fevals
    lookbacks = [1]
    reduced_z = z.copy()
    # These parameters depend on the bitstring size, or feval limit, and hence
    # cannot be homogenized across kernels/feval limit bins.
    forbidden_pars = ['pop_size', 'k', 'nr_particles']
    homo_z = []
    for fevbin in reduced_z:
        bin_lst = []
        for xtup in fevbin:
            hyperpar_str = xtup[0].split(";")
            if len(hyperpar_str) > 1:
                for i in range(1, len(hyperpar_str)):
                    hyperpar_str[i] = hyperpar_str[i][1:]
            reduced_str = []
            for par in hyperpar_str:
                if par.split('=')[0] in forbidden_pars:
                    continue
                reduced_str.append(par)
            bin_lst.append(";".join(reduced_str))
        homo_z.append(bin_lst)

    for l in lookbacks:
        for s in range(l, len(homo_z)):
            if len(homo_z[s]) > 1:
                attempt = list(set(homo_z[s-l]).intersection(set(homo_z[s])))
                if len(attempt) > 0:
                    homo_z[s] = attempt
                    homo_z[s-l] = attempt
        # Forward pass
        for s in range(0, len(homo_z)-l):
            if len(homo_z[s]) > 1:
                attempt = list(set(homo_z[s+l]).intersection(set(homo_z[s])))
                if len(attempt) > 0:
                    homo_z[s] = attempt
                    homo_z[s+l] = attempt


    for j in range(len(homo_z)):
        new_lst = []
        for parvec in homo_z[j]:
            new_tup = tuple(parvec.split(';'))
            new_lst.append(new_tup)
        homo_z[j] = new_lst

    # Now keep only the entries where the non-forbidden vars are in the
    #  homogenized dict
    for fi in range(len(reduced_z)):
        for var_str in reduced_z[fi]:
            varvec = var_str[0].split(';')
            if len(varvec) > 1:
                for k in range(1, len(varvec)):
                    varvec[k] = varvec[k][1:]
            in_homo_dict = False
            for homo_entry in homo_z[fi]:
                # check if this homo entry coincides with varvec
                coincides = True
                for xvar in varvec:
                    if xvar.split('=')[0] in forbidden_pars:
                        continue
                    if xvar not in homo_entry:
                        coincides = False
                if coincides:
                    in_homo_dict = True
                    break
            if not in_homo_dict:
                reduced_z[fi].remove(var_str)
    return z

def find_common_settings(kernels, margin, dir_files, maxfevals):
    common_settings = [[] for x in range(len(maxfevals))]
    sets = []
    filled_bins = []
    for kernel in kernels:
        print("\nProcessing kernel", kernel, "...")
        datafiles = find_data_file(algorithm, kernel, dir_files)
        for datafile in datafiles:
            hyper_data = read_process_file(dir_files, datafile)
            feval_parts, foundbools = partition_feval(hyper_data, maxfevals)
            best_settings = get_best_settings(feval_parts, margin, maxfevals)
            sets.append(best_settings)
            filled_bins.append(foundbools)

    x = []
    for k in range(len(sets[0])):
        if filled_bins[0][k]:
            x.append(sets[0][k])
        else:
            x.append([])

    ### if a bin is filled for some kernel, but not for others, choose the settings
    ### of the bin that does use it as the other kernels will not access these anyway
    for i in range(1, len(sets)):
        binboolsold = filled_bins[i-1]
        binboolsnew = filled_bins[i]
        y = sets[i]
        z = []
        for k in range(len(y)):
            if not binboolsold[k]:# Previous had no relevant settings:
                if binboolsnew[k]:
                    zk = y[k]
                else:
                    zk = []
            elif binboolsnew[k]:# There are relevant settings here
                zk = list(set(x[k]).intersection(set(y[k])))
            else:
                zk = x[k]
            z.append(zk)
        x = z
    for x in range(len(z)):#Make it clear that these are irrelevant bins
        binrelevant = False
        for bbins in filled_bins:
            if bbins[x]:
                binrelevant = True
        if not binrelevant:
            z[x] = [None]

    # Unify the sets so that they are same settings if possible
#    for t in range(30):
#        z = homogenize_dict(z)
    limit = 20
    for x in range(len(z)):
        if len(z[x]) > limit:
            print(maxfevals[x], "of length", len(z[x]), "limited to", limit)
            z[x] = z[x][:limit]
    return z

if __name__ == '__main__':
    ## Which algorithms to run
    #maxfevals = [50,100,150,200,400,600,800,1000,2000]
    maxfevals = [25,50,100,200,400,800,1600]
    algos = [("GLS", 0.02),#DONE
        ("DifferentialEvolution", 0.018),#TODO: REMOVE ITERATIONS AS SPECIFIED LINE 98
        ("BasinHopping", 0.16),
        ("DualAnnealing", 0.02),#DONE
        ("RandomGreedyMLS", 0.04),#4 DONE
        ("BestMLS", 0.05),#DONE
        ("RandomGreedyILS", 0.025),#DONE
        ("BestILS", 0.001),#DONE
        ("RandomGreedyTabu", 0.03),#8 DONE
        ("BestTabu", 0.05),#DONE
        ("ParticleSwarm", 0.012),#DONE
        ("GeneticAlgorithm", 0.055),#DONE
        ("SimulatedAnnealing", 0.13),#12 DONE
    ]
    #NOTE: Choose different limits for different Fevals to get best possible params

    algorithm, margin = algos[0]
    print("\nSearching for optimal settings for", algorithm, "Margin", margin)

    ### Get the files
    # We do hyperparameter tuning on GTX 1080Ti files
    kernels = ['pnpoly', 'GEMM','convolution']

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-1]) + "/"
    file_dir = root_dir + 'GPU_benchmarking_paper/tune_hyperpars_data/'
    settings = find_common_settings(kernels, margin, file_dir, maxfevals)

    import pprint
    print("\n##############")
    for s in range(len(settings)):
        if len(settings[s]) == 0:
            print("\n->->->Margin too small, no match found")
        print("For function evals <=", maxfevals[s], ":")
        pprint.pprint(settings[s])
