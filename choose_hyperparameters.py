import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import warnings

def find_data_file(algo, kernelname, experiment_dir):
    exper_files = [f for f in os.listdir(experiment_dir) if os.path.isfile(os.path.join(experiment_dir, f))]
    data_file = None
    for f in exper_files:
        sls = f.split("_")
        if len(sls) < 4:
            continue
        elif sls[2] == algo:
            if sls[3] == kernelname:
                data_file = f
    if data_file is None:
        raise Exception("Could not find file for this algorithm.")
    return data_file

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

def partition_feval(hyperdata):
    # Partition in right max_feval category
    maxfevals = [50,100,150,200,400,600,800,1000,2000,1000000]#One extra for overflow
    fevalpartition = [[] for x in range(len(maxfevals))]
    idx = 0
    error = 0.05
    found_entries = []# Some algorithms never use many fevals, check if bin is relevant
    for i in range(len(maxfevals)):
        found = False
        for xs in hyperdata:
            if xs[4] <= (1+error)*maxfevals[i]:
                fevalpartition[i].append(xs)
                if i > 0:
                    if xs[4] > (1+error)*maxfevals[i-1]:
                        found = True
                else:
                    found = True
        if not found:
            print("No elements in this partition:", maxfevals[i])
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

def get_best_settings(fevalpartition, margin):
    # Select best settings
    maxfevals = [50,100,150,200,400,600,800,1000,2000,1000000]#One extra for overflow
    fevalsettings = [[] for x in range(len(maxfevals)-1)]
    for i in range(len(maxfevals)-1):
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
            #devoent = entry[0].split(";")[:-1]
            #devoentry = tuple(["".join(devoent)])
            #entry = devoentry
            fevalsettings[i].append(entry)#Need hashable for sets later
    return fevalsettings

def homogenize_dict(z):
    #If there are many settings, try to find those that are similar across fevals
    lookbacks = [1,2,3,4,5,6,7,8]
    for l in lookbacks:
        for s in range(l, len(z)):
            if len(z[s]) > 1:
                attempt = list(set(z[s-l]).intersection(set(z[s])))
                if len(attempt) > 0:
                    z[s] = attempt
                    z[s-l] = attempt
        # Forward pass
        for s in range(0, len(z)-l):
            if len(z[s]) > 1:
                attempt = list(set(z[s+l]).intersection(set(z[s])))
                if len(attempt) > 0:
                    z[s] = attempt
                    z[s+l] = attempt
    return z

def find_common_settings(kernels, margin, dir_files):
    maxfevals = [50,100,150,200,400,600,800,1000,2000]
    common_settings = [[] for x in range(len(maxfevals))]    
    sets = []
    filled_bins = []
    for kernel in kernels:
        print("\nProcessing kernel", kernel, "...")
        datafile = find_data_file(algorithm, kernel, dir_files)
        hyper_data = read_process_file(dir_files, datafile)
        feval_parts, foundbools = partition_feval(hyper_data)
        best_settings = get_best_settings(feval_parts, margin)
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
    #for t in range(100):
    #    z = homogenize_dict(z)
    for x in range(len(z)):
        if len(z[x]) > 10:
            print(maxfevals[x], "of length", len(z[x]), "limited to 10.")
            z[x] = z[x][:10]
    return z

if __name__ == '__main__':
    ## Which algorithms to run
    maxfevals = [50,100,150,200,400,600,800,1000,2000]
    algos = [("GLS", 0.02),
        ("DifferentialEvolution", 0.039),#TODO: REMOVE ITERATIONS AS SPECIFIED LINE 98
        ("BasinHopping", 0.025),
        ("DualAnnealing", 0.003),
        ("RandomGreedyMLS", 0.05),
        ("BestMLS", 0.02),
        ("RandomGreedyILS", 0.0),
        ("BestILS", 0.005),
        ("RandomGreedyTabu", 0.04),
        ("BestTabu", 0.01),
        ("ParticleSwarm", 0.005),
        ("GeneticAlgorithm", 0.007),
        ("SimulatedAnnealing", 0.005),
    ]
    #NOTE: Choose different limits for different Fevals to get best possible params

    algorithm, margin = algos[3]
    print("\nSearching for optimal settings for", algorithm, "Margin", margin)

    ### Get the files
    # We do hyperparameter tuning on GTX 1080Ti files
    kernels = ['pnpoly', 'GEMM','convolution']

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-1]) + "/"
    file_dir = root_dir + 'GPU_benchmarking_paper/tune_hyperpars_data/'
    settings = find_common_settings(kernels, margin, file_dir)
    
    import pprint
    print("\n##############")
    for s in range(len(settings)):
        if len(settings[s]) == 0:
            print("\n->->->Margin too small, no match found")
        print("For function evals <=", maxfevals[s], ":")
        pprint.pprint(settings[s])
