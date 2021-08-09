import numpy as np
import csv
import random
import json
import statistics
import math
import os
import itertools

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
    filenames = ['convolution_P100_processed.json', 'GEMM_P100_processed.json', 'pnpoly_P100_processed.json']
    for filename in filenames:
        with open(data_path + filename, 'r') as myfile:
            data=myfile.read()
        data = json.loads(data)

        print("Device: " + str(data['device_name']))
        print("Kernel name: " + str(data['kernel_name']))
        print("Tunable parameters: " + str(data['tune_params_keys']), end='\n\n')

        # Pre-process the search space
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

        ## Define experimental parameters
        maxtime = 3
        maxfevals = [50,100,150,200,400,600,800,1000,2000]
        minvar = 0.00001
        exper_runs = 50

        #NOTE: To log results, set LOG_results to True
        LOG_RESULTS = False

        ## Which algorithms to run
        allruns = False
        if allruns:
            BASH = True
            DSA = True
            DEVO = True
            PSO = True
            SAN = True
            GLS = True
            GA = True
            GTABU = True
            BTABU = True
            GMLS = True
            BMLS = True
            GILS = True
            BILS = True
        else:
            BASH = False
            DSA = False
            DEVO = False
            PSO = False
            SAN = False
            GLS = False
            GA = False
            GTABU = False
            BTABU = False
            GMLS = False
            BMLS = False
            GILS = False
            BILS = False

        GILS = True

        # Basin Hopping
        if BASH:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]


            # Hyper parameters
            hyperpars = dict()
            hyperpars['method'] = ['Powell','BFGS','COBYLA','L-BFGS-B','SLSQP','CG','Nelder-Mead']
            hyperpars['temperature'] = [0.005, 0.1, 0.25, 0.5, 1.0, 2.0]
           
            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")
           
            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                    results = [[],[]]
                    for i in range(exper_runs):
                        method = combi[0]
                        iterations = 1000000
                        temperature = combi[1]

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
                    experiment_results.append(["Basin hopping", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("\nBasin Hopping: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_BasinHopping_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### Dual Annealing
        if DSA:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

            # Hyper parameters
            hyperpars = dict()
            hyperpars['method'] = ['COBYLA','L-BFGS-B','SLSQP','CG','Powell','Nelder-Mead', 'BFGS', 'trust-constr']
           
            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")

            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                    results = [[],[]]
                    for i in range(exper_runs):
                        # Hyper parameters
                        iterations = 1000000
                        method = combi[0]

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

                    settings = "method=" + method
                    success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                    if statistics.mean(results[1]) > 1.05*maxfeval:
                        print("CHECK THIS THOROUGHLY!")
                        continue
                    experiment_results.append(["Dual annealing", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("\nDual SA: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_DualAnnealing_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### Particle Swarm Optimization (PSO)
        if PSO:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

            # Hyper parameters
            hyperpars = dict()
            # We save the multiplies of maxfeval
            hyperpars['nr_parts'] = [1.0, 0.5, 0.33, 0.2, 0.1, 0.07, 0.03]
            # We save the multiplies which is times nr_particles
            hyperpars['kvar'] = [0.5, 0.33, 0.2, 0.1]
            hyperpars['scaling'] = [10.0, 100.0, 1000.0]
           
            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")

            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                    results = [[],[]]
                    for i in range(exper_runs):
                        # Hyper parameters (there are many more, but won't optimize all of them)
                        nr_parts = max(1, int(combi[0] * maxfeval))
                        kvar = max(1, int(nr_parts * combi[1])) # Should be less than nr_parts
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
                        #iterations = maxfeval//nr_parts+1 #This is how fevals is determined here
                        #if maxfeval <= 200:
                        #    iterations -= 1
                        #if maxfeval >= 800:
                        #    iterations += 1
                        iterations = 100

                        x = test_pso.solve(max_iter=iterations,
                                    max_funcevals=maxfeval,
                                    n_procs=nprocs)
                        results[0].append(best_fit/float(x[0]))
                        results[1].append(x[2])

                    if statistics.mean(results[1]) > 1.05*maxfeval:
                        print("CHECK THIS THOROUGHLY!")
                        continue
                    settings = "nr_particles=" + str(nr_parts) + "; k=" + str(kvar) + "; scaling=" + str(scaling)
                    success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                    experiment_results.append(["PSO", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("PSO: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_ParticleSwarm_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### Differential Evolution
        if DEVO:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

            # Hyper parameters
            hyperpars = dict()
            # We save the multiplies of maxfeval
            hyperpars['pop_size'] = [0.02, 0.04, 0.06, 0.08, 0.1, 0.18, 0.25]
            hyperpars['mutation'] = [(1.0, 1.9), (0.5, 1.9), (0.5, 1.0), (0.2, 0.7)]
            hyperpars['recombination'] = [0.2, 0.5, 0.7, 0.9]
            hyperpars['method'] = ["best1bin",
                "best1exp",
                "rand1exp",
                "randtobest1exp",
                "currenttobest1exp",
                "best2exp",
                "rand2exp",
                "randtobest1bin",
                "currenttobest1bin",
                "best2bin",
                "rand2bin",
                "rand1bin"]
           
            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")
            
            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), int(combi[0]*maxfeval), combi[1:], maxfeval)

                    # Hyper parameters
                    # Max func evals is (maxiter + 1) * popsize * len(x)
                    # so maxiter = maxfeval/(pop_size * nr_vars) - 1
                    pop_size = max(1, int(combi[0] * maxfeval))
                    mutate = combi[1]
                    recomb = combi[2]
                    methd = combi[3]

                    #iterations = max(0, int(maxfeval / (pop_size * nr_vars))-1)
                    iterations = 1000000
                    hc = False # Don't use if you want accuract F_Evals measurements.
                    if pop_size == 1 and methd in ['rand2exp','rand2bin']:
                        # These methods do not support pop_size of 1!
                        continue
                    results = [[],[]]
                    for i in range(exper_runs):
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

                    #if statistics.mean(results[1]) > 1.05*maxfeval:
                    #    print("CHECK THIS THOROUGHLY!")
                    #    continue

                    settings = "pop_size=" + str(pop_size) + "; method=" + str(methd) + "; mutation=" + str(mutate) + "; recombination=" + str(recomb) + "; iterations=" + str(iterations) 
                    success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                    experiment_results.append(["Differential evolution", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("Diff Evo: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_DifferentialEvolution_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### Simulated Annealing
        if SAN:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

            # Hyper parameters
            hyperpars = dict()
            hyperpars['hillclimb'] = [None, hill.BestHillclimb, hill.RandomGreedyHillclimb, hill.StochasticHillclimb]
            hyperpars['explore'] = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
            hyperpars['nbour_method'] = ["Hamming","adjacent"]
            
           
            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")

            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                    results = [[],[]]

                    hillclimber = combi[0]
                    explr = combi[1]
                    explore = int(explr * bsize)
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
                    experiment_results.append(["Simulated annealing", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("SA: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_SimulatedAnnealing_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### GLS
        if GLS:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

            # Hyper parameters
            hyperpars = dict()
            # We save the multiplies of maxfeval
            hyperpars['pop_size'] = [0.025, 0.05, 0.08, 0.12, 0.2, 0.33, 0.5]
            hyperpars['hillclimber'] = [hill.BestHillclimb, hill.RandomGreedyHillclimb, hill.StochasticHillclimb]
            hyperpars['selector'] = [sel.RTS, sel.select_best_half, sel.tournament2_selection, sel.tournament4_selection, sel.tournament8_selection]
            hyperpars['reproductor'] = [rep.uniform_crossover, rep.twopoint_crossover, rep.onepoint_crossover]
            hyperpars['nbour_method'] = ["Hamming","adjacent"]
           
            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")

            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                    # Hyper parameters
                    population_size = max(2, 2*(int(combi[0] * maxfeval)//2))#Must be even
                    hillclimber = combi[1]
                    selector = combi[2]
                    reproductor = combi[3]
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

                    settings = "pop_size=" + str(population_size) + "; reproductor=" + str(reproductor) + "; hillclimber=" + str(hillclimber) + "; selector=" + str(selector) + "; nbour_method=" + str(nbour)
                    success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                    experiment_results.append(["Genetic local search", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("GLS: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1]))) 

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_GLS_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### GA
        if GA:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

            # Hyper parameters
            hyperpars = dict()
            # We save the multiplies of maxfeval
            hyperpars['pop_size'] = [0.025, 0.05, 0.08, 0.12, 0.2, 0.33, 0.5]
            hyperpars['selector'] = [sel.RTS, sel.select_best_half, sel.tournament2_selection, sel.tournament4_selection, sel.tournament8_selection]
            hyperpars['reproductor'] = [rep.uniform_crossover, rep.twopoint_crossover, rep.onepoint_crossover]
            hyperpars['mutation'] = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
           
            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")

            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                    # Hyper parameters
                    population_size = max(2, 2*(int(combi[0] * maxfeval)//2))#Must be even
                    selector = combi[1]
                    reproductor = combi[2]
                    mutate = combi[3]

                    results = run_experiment_ga(exper_runs, population_size, bsize, disc_space.fitness, reproductor, selector, mutate, best_fit, searchspace, maxtime, maxfeval, minvar)

                    settings = "pop_size=" + str(population_size) + "; reproductor=" + str(reproductor) + "; selector=" + str(selector) + "; mutation=" + str(mutate)
                    success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                    experiment_results.append(["Genetic algorithm", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("GA: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1]))) 

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_GeneticAlgorithm_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### Best improvement Tabu search
        if BTABU:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

            # Hyper parameters
            hyperpars = dict()
            hyperpars['tabu_size'] = [4, 8, 16, 32, 64, 100, 200, 400, 700, 1000, 2000]
            hyperpars['nbour_method'] = ["Hamming","adjacent"]
           
            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")

            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                    results = [[],[]]
                    for i in range(exper_runs):
                        tabu_size = combi[0]
                        nbour = combi[1]
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

                    settings = "tabu_size=" + str(tabu_size) + "; nbour=" + str(nbour)
                    success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                    experiment_results.append(["Tabu search", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("BestTabu: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))
     
            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_BestTabu_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)

        ### Greedy Tabu search
        if GTABU:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

            # Hyper parameters
            hyperpars = dict()
            hyperpars['tabu_size'] = [4, 8, 16, 32, 64, 100, 200, 400, 700, 1000, 2000]
            hyperpars['nbour_method'] = ["Hamming","adjacent"]
           
            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")

            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                    results = [[],[]]
                    for i in range(exper_runs):
                        tabu_size = combi[0]
                        nbour = combi[1]
                        test_tabu = tabu.RandomGreedyTabu(disc_space.fitness,
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

                    settings = "tabu_size=" + str(tabu_size) + "; nbour=" + str(nbour)
                    success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                    experiment_results.append(["Tabu search", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("RandomGreedyTabu: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))
     
            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_RandomGreedyTabu_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)

        ### BestMLS
        if BMLS:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

            # Hyper parameters
            hyperpars = dict()
            hyperpars['nbour_method'] = ["Hamming","adjacent"]
           
            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")

            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                    results = [[],[]]
                    for i in range(exper_runs):
                        nbour = combi[0]
                        test_mls = mls.BestMLS(disc_space.fitness,
                                bsize,
                                -1,
                                searchspace=searchspace,
                                neighbour=nbour)

                        iterations = 100000
                        x = test_mls.solve(iterations,
                                    max_time=maxtime,#seconds
                                    stopping_fitness=best_fit,
                                    max_funcevals=maxfeval,
                                    verbose=False)
                        results[0].append(best_fit/float(x[0]))
                        results[1].append(x[2])

                    settings = "nbour=" + str(nbour)
                    success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                    experiment_results.append(["BestMLS", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("BestMLS: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_BestMLS_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### GreedyMLS
        if GMLS:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

            # Hyper parameters
            hyperpars = dict()
            hyperpars['restart'] = [False, True]
            hyperpars['nbour_method'] = ["Hamming","adjacent"]
           
            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")

            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                    results = [[],[]]
                    for i in range(exper_runs):
                        restart = combi[0]
                        nbour = combi[1]
                        test_mls = mls.RandomGreedyMLS(disc_space.fitness,
                                bsize,
                                -1,
                                searchspace=searchspace,
                                neighbour=nbour,
                                restart_search=restart)

                        iterations = 100000
                        x = test_mls.solve(iterations,
                                    max_time=maxtime,#seconds
                                    stopping_fitness=best_fit,
                                    max_funcevals=maxfeval,
                                    verbose=False)
                        results[0].append(best_fit/float(x[0]))
                        results[1].append(x[2])

                    settings = "restart=" + str(restart) + "; nbour=" + str(nbour)
                    success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                    experiment_results.append(["RandomGreedyMLS", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("RandomGreedyMLS: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_RandomGreedyMLS_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)


        ### GreedyILS
        if GILS:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

            # Hyper parameters
            hyperpars = dict()
            hyperpars['walksize'] = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]
            hyperpars['noimprove'] = [5, 10, 25, 50, 100, 200, 500]
            hyperpars['restart'] = [False, True]
            hyperpars['nbour_method'] = ["Hamming","adjacent"]

            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")

            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                    results = [[],[]]
                    for i in range(exper_runs):
                        walksize = combi[0]
                        wsize = int(walksize * bsize)
                        noimp = combi[1]
                        restart = combi[2]
                        nbour = combi[3]
                        test_ils = ils.RandomGreedyILS(disc_space.fitness,
                                bsize,
                                -1,
                                wsize,
                                noimprove=noimp,
                                searchspace=searchspace,
                                neighbour=nbour,
                                restart_search=restart)

                        iterations = 100000
                        x = test_ils.solve(iterations,
                                    max_time=maxtime,#seconds
                                    stopping_fitness=best_fit,
                                    max_funcevals=maxfeval,
                                    verbose=False)
                        results[0].append(best_fit/float(x[0]))
                        results[1].append(x[2])
             
                    settings = "walksize=" + str(walksize) + "; no_improve=" + str(noimp) + "; restart=" + str(restart) + "; nbour=" + str(nbour)
                    success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                    experiment_results.append(["RandomGreedyILS", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("RandomGreedyILS: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_RandomGreedyILS_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)

        ### BestILS
        if BILS:
            experiment_results = [[filename[:-5]],["Algorithm", "Mean fraction of optimum", "StDev fraction of optimum", "Success rate", "Mean function evaluations", "StDev function evaluations", "Settings"]]

            # Hyper parameters
            hyperpars = dict()
            hyperpars['walksize'] = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]
            hyperpars['noimprove'] = [5, 10, 25, 50, 100, 200, 500]
            hyperpars['nbour_method'] = ["Hamming","adjacent"]

            hp_combis = list(itertools.product(*list(hyperpars.values())))
            print("There are", len(maxfevals) * len(hp_combis), "combinations to tune...")

            hprun = 0
            for maxfeval in maxfevals:
                for combi in hp_combis:
                    hprun += 1
                    print("\n","Commencing run {0}...".format(hprun), combi, maxfeval)
                    results = [[],[]]
                    for i in range(exper_runs):
                        walksize = combi[0]
                        wsize = int(walksize * bsize)
                        noimp = combi[1]
                        nbour = combi[2]
                        test_ils = ils.BestILS(disc_space.fitness,
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
             
                    settings = "walksize=" + str(walksize) + "; no_improve=" + str(noimp) + "; nbour=" + str(nbour)
                    success_rate = (np.array(results[0]) == 1.0).sum()/float(exper_runs)
                    experiment_results.append(["BestILS", statistics.mean(results[0]), statistics.stdev(results[0]), success_rate, statistics.mean(results[1]), statistics.stdev(results[1]), settings])

                    print("BestILS: Average fraction of optimal fitness: {0:.4f} +- {1:.5f}".format(statistics.mean(results[0]), statistics.stdev(results[0])), "\nAverage number of function evaluations: {0:.4f} +- {1:.5f}".format(statistics.mean(results[1]), statistics.stdev(results[1])))

            ### Write results to file
            if LOG_RESULTS:
                export_filename = "tune_hyperpars_BestILS_" + filename[:-5] + "_runs={0}".format(exper_runs) + ".csv"
                with open(export_filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(experiment_results)
