from deap import base, creator
import random
from deap import tools
import numpy as np
import matplotlib.pyplot as plt


# statistics registeration
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
#stats.register("avg", np.mean)
#stats.register("std", np.std)
stats.register("min", np.min)
#stats.register("max", np.max)

def run_ga(pool = None, CXPB = 0.9, MUTPB = 0.2, LSPB = 0.2, NGEN = 100, ind_type = np.ndarray,
           ind_size = None, pop_size = 50, ind_gen = None, mate = None,
           mutate = None, select = tools.selTournament, evaluate = None, local_search = None):
           
    toolbox = base.Toolbox()
    if pool != None:
        toolbox.register("map", pool.map)
           
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", ind_type, fitness=creator.FitnessMin)
    
    
    toolbox.register("attribute", ind_gen)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    pop = toolbox.population(n=pop_size)
    
    toolbox.register("mate", mate) #tools.cxTwoPoint)
    toolbox.register("mutate", mutate, indpb=0.1)#tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("local_search", local_search)
    toolbox.register("select", select, tournsize=len(pop)/4)
    toolbox.register("evaluate", evaluate)

    
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    min_fit_lst = []
    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        for indvidual in offspring:
            if random.random() < LSPB:
                toolbox.local_search(indvidual)
                del indvidual.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # printing statistics
        record = stats.compile(pop)
        min_fit = record['min']
        min_fit_lst.append(min_fit)
        print "gen #%d: stats:%f"%(g, min_fit)
    plt.plot(min_fit_lst)
    plt.show()
    return pop