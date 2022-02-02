import numpy as np
import csv
import random
import json
import statistics
import math
import os

import bloopy.utils as utils
import bloopy.reproductive_functions as rep
import bloopy.selection_functions as sel

import gpu_utils

import bloopy.algorithms.genetic_algorithm as ga
import bloopy.algorithms.local_search as mls
import bloopy.algorithms.iterative_local_search as ils
import bloopy.algorithms.tabu_search as tabu
import bloopy.algorithms.genetic_local_search as gls
import bloopy.algorithms.simulated_annealing as sa
import bloopy.algorithms.differential_evolution as de
import bloopy.algorithms.hillclimbers as hill
import bloopy.algorithms.dual_annealing as dsa
import bloopy.algorithms.pso_pyswarms as psop
import bloopy.algorithms.basin_hopping as bashop
import bloopy.algorithms.random_sampling as randsl


def run_experiment_ga(iters, population_size, bsize, ffunc, reproductor, selector, mutation, best_fit, sspace, maxtime, maxfeval, minvar):
    frac_optim = []
    func_evals = []
    for it in range(iters):
        # Need to generate own population, else fitness function will catch error
        input_pop = utils.generate_population(population_size, ffunc, sspace)
        test_ga = ga.genetic_algorithm(ffunc,
                    reproductor,
                    selector,
                    population_size,
                    bsize,
                    min_max_problem=-1,
                    searchspace=sspace,
                    input_pop=input_pop,
                    mutation=mutation)

        # Solve with GA
        x = test_ga.solve(min_variance=minvar,
                    max_iter=1000000,
                    no_improve=100000,
                    max_time=maxtime,#seconds
                    stopping_fitness=best_fit,
                    max_funcevals=maxfeval,
                    verbose=False)
        frac_optim.append(best_fit/float(x[0]))
        func_evals.append(x[4])
    return frac_optim, func_evals


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
    #'convolution_P100_processed.json',#tuning
    'convolution_K20_processed.json',
    'convolution_GTX_Titan_X_processed.json']

    GEMM_files = ['GEMM_A100_processed.json',
    'GEMM_V100_processed.json',
    'GEMM_RTX_2070_SUPER_processed.json',
    'GEMM_TITAN_RTX_processed.json',
    'MI50_GEMM_processed.json',
    'GEMM_GTX_1080Ti_processed.json',
    #'GEMM_P100_processed.json',#tuning
    'GEMM_K20_processed.json',
    'GEMM_GTX_Titan_X_processed.json']

    pnpoly_files = ['pnpoly_A100_processed.json',
    'pnpoly_V100_processed.json',
    'pnpoly_RTX_2070_SUPER_processed.json',
    'pnpoly_TITAN_RTX_processed.json',
    'pnpoly_GTX_1080Ti_processed.json',
    #'pnpoly_P100_processed.json',#tuning
    'pnpoly_K20_processed.json',
    'pnpoly_GTX_Titan_X_processed.json']

    for filename in convolution_files:
        with open(data_path + filename, 'r') as myfile:
            data=myfile.read()
        data = json.loads(data)

        print("Device: " + str(data['device_name']))
        print("Kernel name: " + str(data['kernel_name']))
        print("Tunable parameters: " + str(data['tune_params_keys']), end='\n\n')


        ### Pre-process the search space, remove variables with only one possible value
        searchspace_orig = data['tune_params']
        searchspace = utils.clean_up_searchspace(searchspace_orig)
        print("Processed search space:", searchspace)

        ### Calculate bitstring size
        bsize = utils.calculate_bitstring_length(searchspace)
        print("Size of bitstring after pre-processing:", bsize)

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
        print("Optimal settings in cache are:", bestkey, "with time {0:.4f}".format(best_fit))
        print("There are", len(data['cache'].keys()), "keys in the searchspace")

        ## Define experimental parameters
        maxtime = 10
        maxfevals = [25,50,75,100,150,200,400,600,800,1000,2000]
        minvar = 1e-10
        exper_runs = 100
        output_dir = '/experiment_files/'

        #NOTE: To log results, set LOG_results to True
        LOG_RESULTS = False

        ## Which algorithms to run
        allruns = False
        if allruns:
            RANDSAM = True
            GLS = True
            TABU = True
            MLS = True
            ILS = True
            SAN = True
            DSA = True
            GA = True
            PSO = True
            BASH = True
            DEVO = True
        else:
            RANDSAM = False
            BASH = False
            DSA = False
            DEVO = False
            SAN = False
            GLS = False
            GA = False
            TABU = False
            MLS = False
            ILS = False
            PSO = False

        ILS = True

        ### SEED
        #np.random.seed(1234567)
        #random.seed(1234567)

        # Random sampling
        if RANDSAM:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings","MaxFEval"]]
            for maxfeval in maxfevals:
                results = [[],[]]
                for i in range(exper_runs):
                    test_rand = randsl.random_sampling(disc_space.fitness,
                            bsize,
                            -1,
                            searchspace)

                    x = test_rand.solve(max_time=maxtime,#seconds
                                stopping_fitness=best_fit,
                                max_funcevals=maxfeval)
                    results[0].append(best_fit/float(x[0]))
                    results[1].append(x[2])

                success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                experiment_results.append(["Random sampling", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]),"NA",maxfeval])

                print("\nRandom sampling: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])), "\nAverage time to find config (sec): Not possible")

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "RandomSampling_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        # Basin Hopping
        if BASH:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings","MaxFEval"]]

            #Feval settings
            paramsettings = dict()
            paramsettings[25] = ["COBYLA", 0.5]
            paramsettings[50] = ["COBYLA", 0.5]
            paramsettings[75] = ["COBYLA", 1.0]
            paramsettings[100] = ["COBYLA", 1.0]
            paramsettings[150] = ["COBYLA", 0.1]
            paramsettings[200] = ["COBYLA", 0.25]
            paramsettings[400] = ["COBYLA", 0.25]
            paramsettings[600] = ["SLSQP", 2.0]
            paramsettings[800] = ["SLSQP", 0.005]
            paramsettings[1000] = ["SLSQP", 1.0]
            paramsettings[2000] = ["SLSQP", 0.1]

            hprun = 0
            for maxfeval in maxfevals:
                combi = paramsettings[maxfeval]
                print(combi)
                hprun += 1
                print("Commencing run {0}...".format(hprun), maxfeval)
                results = [[],[]]
                for i in range(exper_runs):
                    method = combi[0]
                    temperature = combi[1]
                    iterations = 1000000

                    test_bash = bashop.basin_hopping(disc_space.fitness,
                            -1,
                            searchspace,
                            T=temperature,
                            method=method)

                    x = test_bash.solve(max_iter=iterations,
                                max_time=maxtime,#seconds
                                stopping_fitness=best_fit,
                                max_funcevals=maxfeval)
                    results[0].append(best_fit/float(x[0]))
                    results[1].append(x[2])

                settings = "method=" + method + "; iterations=" + str(iterations) + "; temperature=" + str(temperature)
                success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                experiment_results.append(["Basin hopping", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])

                print("\nBasin Hopping: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "BasinHopping_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### Dual Annealing
        if DSA:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings","MaxFEval"]]

            #Feval settings
            paramsettings = dict()
            paramsettings[25] = ["Powell"]
            paramsettings[50] = ["Powell"]
            paramsettings[75] = ["Powell"]
            paramsettings[100] = ["Powell"]
            paramsettings[150] = ["Powell"]
            paramsettings[200] = ["Powell"]
            paramsettings[400] = ["Powell"]
            paramsettings[600] = ["Powell"]
            paramsettings[800] = ["Powell"]
            paramsettings[1000] = ["Powell"]
            paramsettings[2000] = ["Powell"]

            hprun = 0
            for maxfeval in maxfevals:
                combi = paramsettings[maxfeval]
                hprun += 1
                print("\n","Commencing run {0}...".format(hprun), combi, maxfevals[hprun-1])
                results = [[],[]]
                for i in range(exper_runs):
                    # Hyper parameters
                    method = combi[0]
                    iterations = 1000000

                    test_dsa = dsa.dual_annealing(disc_space.fitness,
                            -1,
                            searchspace,
                            method=method)

                    x = test_dsa.solve(max_iter=iterations,
                                max_time=maxtime,#seconds
                                stopping_fitness=best_fit,
                                max_funcevals=maxfeval)
                    results[0].append(best_fit/float(x[0]))
                    results[1].append(x[2])

                settings = "iterations=" + str(iterations) + "; method=" + method
                success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                experiment_results.append(["Dual annealing", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])

                print("\nDual SA: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "DualAnnealing_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### Particle Swarm Optimization (PSO)
        if PSO:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings","MaxFEval"]]

            #Feval settings
            paramsettings = dict()
            paramsettings[25] = [10, 2, 1000]
            paramsettings[50] = [10, 2, 1000]
            paramsettings[75] = [10, 5, 10]
            paramsettings[100] = [10, 5, 10]
            paramsettings[150] = [15, 7, 1000]
            paramsettings[200] = [20, 10, 1000]
            paramsettings[400] = [28, 9, 100]
            paramsettings[600] = [42, 13, 1000]
            paramsettings[800] = [56, 11, 1000]
            paramsettings[1000] = [100, 10, 1000]
            paramsettings[2000] = [200, 20, 1000]

            hprun = 0
            for maxfeval in maxfevals:
                combi = paramsettings[maxfeval]
                print(combi)
                hprun += 1
                print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                results = [[],[]]
                for i in range(exper_runs):
                    # Hyper parameters (there are many more, but won't optimize all of them)
                    nr_parts = combi[0]
                    kvar = combi[1] # Should be less than nr_parts
                    # Scaled parameters equidistant to [0,scaling],
                    # Strangely, [0,1] is not necessarily the same as e.g. [0,10]
                    scaling = combi[2]

                    test_pso = psop.pyswarms_pso(disc_space.fitness,
                            -1,
                            searchspace,
                            n_particles=nr_parts,
                            k=kvar,
                            scaling=scaling)

                    nprocs = None
                    iterations = 100

                    x = test_pso.solve(max_iter=iterations,
                                max_funcevals=maxfeval,
                                n_procs=nprocs)
                    results[0].append(best_fit/float(x[0]))
                    results[1].append(x[2])

                settings = "nr_particles=" + str(nr_parts) + "; k=" + str(kvar) + "; scaling=" + str(scaling)
                success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                experiment_results.append(["PSO", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])

                print("PSO: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "ParticleSwarm_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### Differential Evolution
        if DEVO:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings","MaxFEval"]]

            #Feval settings
            paramsettings = dict()
            paramsettings[25] = ["best2exp", 1, 0.9, (0.2,0.7)]
            paramsettings[50] = ["best2exp", 1, 0.9, (0.2,0.7)]
            paramsettings[75] = ["best2exp", 2, 0.7, (0.2,0.7)]
            paramsettings[100] = ["best2exp", 2, 0.7, (0.2,0.7)]
            paramsettings[150] = ["best2exp", 3, 0.9, (0.2,0.7)]
            paramsettings[200] = ["best2exp", 4, 0.9, (0.2,0.7)]
            paramsettings[400] = ["randtobest1bin", 8, 0.9, (0.2,0.7)]
            paramsettings[600] = ["best1bin", 12, 0.7, (0.2,0.7)]
            paramsettings[800] = ["best1bin", 16, 0.5, (0.2,0.7)]
            paramsettings[1000] = ["best1bin", 20, 0.5, (0.2,0.7)]
            paramsettings[2000] = ["best1bin", 40, 0.9, (0.5,1.0)]

            hprun = 0
            for maxfeval in maxfevals:
                combi = paramsettings[maxfeval]
                print(combi)
                hprun += 1
                print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                results = [[],[]]
                for i in range(exper_runs):
                    # Hyper parameters
                    # Max func evals is (maxiter + 1) * popsize * len(x)
                    # so maxiter = maxfeval/(pop_size * nr_vars) - 1
                    methd = combi[0]
                    pop_size = combi[1]
                    recomb = combi[2]
                    mutate = combi[3]

                    iterations = max(0, int(maxfeval / (pop_size * nr_vars))-1)
                    if maxfeval >= 200:
                        iterations += 1
                    if maxfeval >= 800:
                        iterations += 1
                    iterations += 100

                    hc = False # For accurate Feval measurements

                    test_diffevo = de.differential_evolution(disc_space.fitness,
                            -1,
                            searchspace,
                            method=methd,
                            mutation=mutate,
                            recombination=recomb,
                            hillclimb=hc,
                            pop_size=pop_size)

                    x = test_diffevo.solve(min_variance=minvar,
                                max_iter=iterations,
                                max_time=maxtime,#seconds
                                stopping_fitness=best_fit,
                                max_funcevals=maxfeval)
                    results[0].append(best_fit/float(x[0]))
                    results[1].append(x[2])

                settings = "pop_size=" + str(pop_size) + "; method=" + str(methd) + "; mutation=" + str(mutate) + "; recombination=" + str(recomb) + "; iterations=" + str(iterations)
                success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                experiment_results.append(["Differential evolution", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])

                print("Diff Evo: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])), "\nAverage time to find config (sec): Not possible")

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "DifferentialEvolution_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### Simulated Annealing
        if SAN:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings","MaxFEval"]]

            #Feval settings
            paramsettings = dict()
            paramsettings[25] = [1.0, hill.RandomGreedyHillclimb, "Hamming"]
            paramsettings[50] = [1.0, hill.RandomGreedyHillclimb, "Hamming"]
            paramsettings[75] = [0.5, hill.RandomGreedyHillclimb, "Hamming"]
            paramsettings[100] = [0.5, hill.RandomGreedyHillclimb, "Hamming"]
            paramsettings[150] = [1.0, hill.RandomGreedyHillclimb, "Hamming"]
            paramsettings[200] = [0.7, hill.RandomGreedyHillclimb, "Hamming"]
            paramsettings[400] = [0.5, hill.RandomGreedyHillclimb, "Hamming"]
            paramsettings[600] = [0.5, hill.RandomGreedyHillclimb, "Hamming"]
            paramsettings[800] = [0.7, hill.RandomGreedyHillclimb, "Hamming"]
            paramsettings[1000] = [0.5, hill.BestHillclimb, "Hamming"]
            paramsettings[2000] = [0.5, hill.BestHillclimb, "Hamming"]

            hprun = 0
            for maxfeval in maxfevals:
                combi = paramsettings[maxfeval]
                print(combi)
                hprun += 1
                print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                results = [[],[]]

                explr = combi[0]
                explore = int(explr * bsize)
                hillclimber = combi[1]
                nbour = combi[2]
                for i in range(exper_runs):
                    test_sa = sa.simulated_annealing(disc_space.fitness,
                            bsize,
                            -1,
                            explore,
                            hillclimb=hillclimber,
                            searchspace=searchspace,
                            neighbour=nbour)

                    iterations = 10000
                    x = test_sa.solve(iterations,
                                max_time=maxtime,#seconds
                                stopping_fitness=best_fit,
                                max_funcevals=maxfeval,
                                verbose=False)
                    results[0].append(best_fit/float(x[0]))
                    results[1].append(x[2])

                settings = "hillclimb=" + str(hillclimber) + "; explore=" + str(explore) + "; nbour_method=" + str(nbour)
                success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                experiment_results.append(["Simulated annealing", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])

                print("SA: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "SimulatedAnnealing_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### GLS
        if GLS:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings","MaxFEval"]]

            #Feval settings
            paramsettings = dict()
            paramsettings[25] = [hill.RandomGreedyHillclimb, 4, rep.onepoint_crossover, sel.select_best_half, 'Hamming']
            paramsettings[50] = [hill.RandomGreedyHillclimb, 4, rep.onepoint_crossover, sel.select_best_half, 'Hamming']
            paramsettings[75] = [hill.RandomGreedyHillclimb, 4, rep.twopoint_crossover, sel.select_best_half, 'Hamming']
            paramsettings[100] = [hill.RandomGreedyHillclimb, 4, rep.twopoint_crossover, sel.select_best_half, 'Hamming']
            paramsettings[150] = [hill.RandomGreedyHillclimb, 6, rep.uniform_crossover, sel.select_best_half, 'Hamming']
            paramsettings[200] = [hill.RandomGreedyHillclimb, 4, rep.twopoint_crossover, sel.tournament8_selection, 'Hamming']
            paramsettings[400] = [hill.RandomGreedyHillclimb, 10, rep.onepoint_crossover, sel.tournament4_selection, 'Hamming']
            paramsettings[600] = [hill.RandomGreedyHillclimb, 20, rep.uniform_crossover, sel.select_best_half, 'adjacent']
            paramsettings[800] = [hill.RandomGreedyHillclimb, 20, rep.onepoint_crossover, sel.RTS, 'adjacent']
            paramsettings[1000] = [hill.RandomGreedyHillclimb, 24, rep.uniform_crossover, sel.RTS, 'adjacent']
            paramsettings[2000] = [hill.RandomGreedyHillclimb, 50, rep.onepoint_crossover, sel.RTS, 'adjacent']

            hprun = 0
            for maxfeval in maxfevals:
                combi = paramsettings[maxfeval]
                hprun += 1
                print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                # Hyper parameters
                hillclimber = combi[0]
                population_size = combi[1]
                reproductor = combi[2]
                selector = combi[3]
                nbour = combi[4]

                results = [[],[]]
                for i in range(exper_runs):
                    # Need to generate own population, else fitness function will catch error
                    input_pop = utils.generate_population(population_size, disc_space.fitness, searchspace)
                    test_gls = gls.genetic_local_search(disc_space.fitness,
                                reproductor,
                                selector,
                                population_size,
                                bsize,
                                hillclimber,
                                min_max_problem=-1,
                                searchspace=searchspace,
                                input_pop=input_pop,
                                neighbour=nbour)

                    # Solve with GLS
                    iterations = 100000
                    x = test_gls.solve(min_variance=minvar,
                                max_iter=iterations,
                                no_improve=100000,
                                max_time=maxtime,#seconds
                                stopping_fitness=best_fit,
                                max_funcevals=maxfeval,
                                verbose=False)

                    results[0].append(best_fit/float(x[0]))
                    results[1].append(x[4])

                settings = "pop_size=" + str(population_size) + "; reproductor=" + str(reproductor) + "; hillclimber=" + str(hillclimber) + "; selector=" + str(selector) + "; nbour_method" + str(nbour)
                success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                experiment_results.append(["Genetic local search", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])

                print("GLS: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "GLS_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### GA
        if GA:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings","MaxFEval"]]

            #Feval settings
            paramsettings = dict()
            paramsettings[25] = [0.02, 10, rep.onepoint_crossover, sel.tournament8_selection]
            paramsettings[50] = [0.02, 14, rep.onepoint_crossover, sel.tournament8_selection]
            paramsettings[75] = [0.02, 16, rep.onepoint_crossover, sel.tournament8_selection]
            paramsettings[100] = [0.02, 20, rep.onepoint_crossover, sel.tournament8_selection]
            paramsettings[150] = [0.02, 24, rep.twopoint_crossover, sel.select_best_half]
            paramsettings[200] = [0.02, 32, rep.uniform_crossover, sel.tournament8_selection]
            paramsettings[400] = [0.05, 24, rep.twopoint_crossover, sel.select_best_half]
            paramsettings[600] = [0.05, 40, rep.twopoint_crossover, sel.select_best_half]
            paramsettings[800] = [0.05, 50, rep.twopoint_crossover, sel.select_best_half]
            paramsettings[1000] = [0.05, 160, rep.twopoint_crossover, sel.tournament4_selection]
            paramsettings[2000] = [0.05, 160, rep.twopoint_crossover, sel.select_best_half]

            hprun = 0
            for maxfeval in maxfevals:
                combi = paramsettings[maxfeval]
                print(combi)
                hprun += 1
                print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                # Hyper parameters
                mutate = combi[0]
                population_size = combi[1]
                reproductor = combi[2]
                selector = combi[3]

                results = run_experiment_ga(exper_runs, population_size, bsize, disc_space.fitness, reproductor, selector, mutate, best_fit, searchspace, maxtime, maxfeval, minvar)

                settings = "pop_size=" + str(population_size) + "; reproductor=" + str(reproductor) + "; selector=" + str(selector) + "; mutation=" + str(mutate)
                success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                experiment_results.append(["Genetic algorithm", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])

                print("GA: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "GeneticAlgorithm_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### Tabu search
        if TABU:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings","MaxFEval"]]
            TABU_TYPE = "RandomGreedy"
            #TABU_TYPE = "Best"

            #Feval settings
            paramsettings = dict()
            if TABU_TYPE == "Best":
                paramsettings[25] = [100, "Hamming"]
                paramsettings[50] = [100, "Hamming"]
                paramsettings[75] = [100, "Hamming"]
                paramsettings[100] = [100, "Hamming"]
                paramsettings[150] = [100, "Hamming"]
                paramsettings[200] = [100, "Hamming"]
                paramsettings[400] = [400, "Hamming"]
                paramsettings[600] = [400, "Hamming"]
                paramsettings[800] = [400, "Hamming"]
                paramsettings[1000] = [400, "Hamming"]
                paramsettings[2000] = [400, "adjacent"]
            elif TABU_TYPE == "RandomGreedy":
                paramsettings[25] = [100, "Hamming"]
                paramsettings[50] = [100, "Hamming"]
                paramsettings[75] = [32, "Hamming"]
                paramsettings[100] = [32, "Hamming"]
                paramsettings[150] = [16, "Hamming"]
                paramsettings[200] = [16, "Hamming"]
                paramsettings[400] = [16, "Hamming"]
                paramsettings[600] = [400, "Hamming"]
                paramsettings[800] = [400, "Hamming"]
                paramsettings[1000] = [400, "Hamming"]
                paramsettings[2000] = [400, "adjacent"]

            hprun = 0
            for maxfeval in maxfevals:
                combi = paramsettings[maxfeval]
                print(combi)
                hprun += 1
                print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                results = [[],[]]
                for i in range(exper_runs):
                    tabu_size = combi[0]
                    nbour = combi[1]
                    if TABU_TYPE == "RandomGreedy":
                        test_tabu = tabu.RandomGreedyTabu(disc_space.fitness,
                                bsize,
                                -1,
                                tabu_size,
                                searchspace=searchspace,
                                neighbour=nbour)
                    elif TABU_TYPE == "Best":
                        test_tabu = tabu.BestTabu(disc_space.fitness,
                                bsize,
                                -1,
                                tabu_size,
                                searchspace=searchspace,
                                neighbour=nbour)

                    iterations = 100000
                    x = test_tabu.solve(iterations,
                                max_time=maxtime,#seconds
                                stopping_fitness=best_fit,
                                max_funcevals=maxfeval,
                                verbose=False)
                    results[0].append(best_fit/float(x[0]))
                    results[1].append(x[2])

                settings = "tabu_size=" + str(tabu_size) + "; iterations=" + str(iterations)
                success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                if TABU_TYPE == "RandomGreedy":
                    experiment_results.append(["RandomGreedyTabu", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])
                elif TABU_TYPE == "Best":
                    experiment_results.append(["BestTabu", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])

                if TABU_TYPE == "RandomGreedy":
                    print("RandomGreedyTabu: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))
                elif TABU_TYPE == "Best":
                    print("BestTabu: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                if TABU_TYPE == "RandomGreedy":
                    export_filename = "RandomGreedyTabu_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                elif TABU_TYPE == "Best":
                    export_filename = "BestTabu_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### MLS
        if MLS:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings","MaxFEval"]]

            #NOTE: Old MLS is OrderedGreedyMLS with Hamming, restart=False, order=None
            MLS_TYPE = "RandomGreedy"
            #MLS_TYPE = "Best"
            #MLS_TYPE = "OrderedGreedy"
            #MLS_TYPE = "Stochastic"

            #Feval settings
            paramsettings = dict()
            if MLS_TYPE == "RandomGreedy":
                paramsettings[25] = ["Hamming", True]
                paramsettings[50] = ["Hamming", True]
                paramsettings[75] = ["Hamming", True]
                paramsettings[100] = ["Hamming", True]
                paramsettings[150] = ["Hamming", False]
                paramsettings[200] = ["Hamming", True]
                paramsettings[400] = ["Hamming", True]
                paramsettings[600] = ["Hamming", False]
                paramsettings[800] = ["Hamming", False]
                paramsettings[1000] = ["Hamming", True]
                paramsettings[2000] = ["Hamming", True]
            if MLS_TYPE == "Best":
                paramsettings[25] = ["adjacent"]
                paramsettings[50] = ["adjacent"]
                paramsettings[75] = ["adjacent"]
                paramsettings[100] = ["adjacent"]
                paramsettings[150] = ["Hamming"]
                paramsettings[200] = ["Hamming"]
                paramsettings[400] = ["Hamming"]
                paramsettings[600] = ["Hamming"]
                paramsettings[800] = ["Hamming"]
                paramsettings[1000] = ["Hamming"]
                paramsettings[2000] = ["Hamming"]

            hprun = 0
            for maxfeval in maxfevals:
                combi = paramsettings[maxfeval]
                nbour = combi[0]
                if "Greedy" in MLS_TYPE:
                    restart = combi[1]
                hprun += 1
                print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                results = [[],[]]
                for i in range(exper_runs):
                    if MLS_TYPE == "RandomGreedy":
                        test_mls = mls.RandomGreedyMLS(disc_space.fitness,
                                bsize,
                                -1,
                                searchspace=searchspace,
                                neighbour='Hamming',
                                restart_search=True)
                    elif MLS_TYPE == "OrderedGreedy":
                        test_mls = mls.OrderedGreedyMLS(disc_space.fitness,
                                bsize,
                                -1,
                                searchspace=searchspace,
                                neighbour='Hamming',
                                restart_search=True,
                                order=None)
                    elif MLS_TYPE == "Best":
                        test_mls = mls.BestMLS(disc_space.fitness,
                                bsize,
                                -1,
                                searchspace=searchspace,
                                neighbour='Hamming')
                    elif MLS_TYPE == "Stochastic":
                        test_mls = mls.StochasticMLS(disc_space.fitness,
                                bsize,
                                -1,
                                searchspace=searchspace,
                                neighbour='adjacent')

                    iterations = 10000
                    x = test_mls.solve(iterations,
                                max_time=maxtime,#seconds
                                stopping_fitness=best_fit,
                                max_funcevals=maxfeval,
                                verbose=False)
                    results[0].append(best_fit/float(x[0]))
                    results[1].append(x[2])

                settings = "iterations=" + str(iterations)
                success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                if MLS_TYPE == "RandomGreedy":
                    experiment_results.append(["RandomGreedyMLS", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])
                elif MLS_TYPE == "Best":
                    experiment_results.append(["BestMLS", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])
                else:
                    raise Exception("Unknown MLS type")

                if MLS_TYPE == "RandomGreedy":
                    print("RandomgGreedyMLS: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))
                elif MLS_TYPE == "Best":
                    print("BestMLS: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                if MLS_TYPE == "RandomGreedy":
                    export_filename = "RandomGreedyMLS_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                elif MLS_TYPE == "Best":
                    export_filename = "BestMLS_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### ILS
        if ILS:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings","MaxFEval"]]

            ILS_TYPE = "RandomGreedy"
            #ILS_TYPE = "Best"
            #ILS_TYPE = "OrderedGreedy"
            #ILS_TYPE = "Stochastic"

            #Feval settings
            paramsettings = dict()
            if ILS_TYPE == "RandomGreedy":
                paramsettings[25] = [0.5, 100, 'Hamming', False]
                paramsettings[50] = [0.5, 100, 'Hamming', False]
                paramsettings[75] = [0.02, 100, 'Hamming', False]
                paramsettings[100] = [0.02, 100, 'Hamming', False]
                paramsettings[150] = [0.5, 10, 'Hamming', False]
                paramsettings[200] = [0.02, 50, 'adjacent', False]
                paramsettings[400] = [0.1, 5, 'adjacent', False]
                paramsettings[600] = [0.05, 5, 'adjacent', True]
                paramsettings[800] = [0.05, 5, 'adjacent', True]
                paramsettings[1000] = [0.05, 5, 'adjacent', True]
                paramsettings[2000] = [0.05, 5, 'adjacent', True]
            elif ILS_TYPE == "Best":
                paramsettings[25] = [0.02, 100, 'adjacent']
                paramsettings[50] = [0.02, 100, 'adjacent']
                paramsettings[75] = [0.02, 25, 'Hamming']
                paramsettings[100] = [0.02, 25, 'Hamming']
                paramsettings[150] = [0.35, 50, 'Hamming']
                paramsettings[200] = [0.35, 10, 'Hamming']
                paramsettings[400] = [0.02, 10, 'adjacent']
                paramsettings[600] = [0.02, 10, 'adjacent']
                paramsettings[800] = [0.02, 10, 'adjacent']
                paramsettings[1000] = [0.02, 10, 'adjacent']
                paramsettings[2000] = [0.02, 10, 'adjacent']

            hprun = 0
            for maxfeval in maxfevals:
                combi = paramsettings[maxfeval]
                print(combi)
                hprun += 1
                print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                walksize = combi[0]
                wsize = int(walksize * bsize)
                noimp = combi[1]
                nbour = combi[2]
                if "Greedy" in ILS_TYPE:
                    restart = combi[3]

                results = [[],[]]
                for i in range(exper_runs):
                    if ILS_TYPE == 'Best':
                        test_ils = ils.BestILS(disc_space.fitness,
                                bsize,
                                -1,
                                wsize,
                                noimprove=noimp,
                                searchspace=searchspace,
                                neighbour=nbour)
                    elif ILS_TYPE == 'RandomGreedy':
                        test_ils = ils.RandomGreedyILS(disc_space.fitness,
                                bsize,
                                -1,
                                wsize,
                                noimprove=noimp,
                                searchspace=searchspace,
                                neighbour=nbour,
                                restart_search=restart)
                    elif ILS_TYPE == 'OrderedGreedy':
                        test_ils = ils.OrderedGreedyILS(disc_space.fitness,
                                bsize,
                                -1,
                                wsize,
                                noimprove=noimp,
                                searchspace=searchspace,
                                neighbour=nbour,
                                restart_search=restart,
                                order=None)
                    elif ILS_TYPE == 'Stochastic':
                        test_ils = ils.StochasticILS(disc_space.fitness,
                                bsize,
                                -1,
                                wsize,
                                noimprove=noimp,
                                searchspace=searchspace,
                                neighbour=nbour)

                    iterations = 100000
                    x = test_ils.solve(iterations,
                                max_time=maxtime,#seconds
                                stopping_fitness=best_fit,
                                max_funcevals=maxfeval,
                                verbose=False)
                    results[0].append(best_fit/float(x[0]))
                    results[1].append(x[2])

                settings = "walksize=" + str(walksize) + "; no_improve=" + str(noimp) + "; iterations=" + str(iterations)
                success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                if ILS_TYPE == "RandomGreedy":
                    experiment_results.append(["RandomGreedyILS", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])
                elif ILS_TYPE == "Best":
                    experiment_results.append(["BestILS", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings, maxfeval])
                else:
                    raise Exception("Unknown ILS type")

                if ILS_TYPE == "RandomGreedy":
                    print("RandomGreedyILS: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))
                elif ILS_TYPE == "Best":
                    print("BestILS: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                if ILS_TYPE == "RandomGreedy":
                    export_filename = "RandomGreedyILS_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                elif ILS_TYPE == "Best":
                    export_filename = "BestILS_" + filename[:-15] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)
