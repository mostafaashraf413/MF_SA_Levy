from deap import base, creator
import random
from deap import tools
import numpy as np

V = [[0,1,0,1,0],
     [1,0,1,0,1],
     [0,1,0,0,0],
     [1,0,0,0,1],
     [0,1,0,1,0]]

#V = [[0,1,0],
#     [1,0,1],
#     [0,1,0]]

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

IND_SIZE = 1#len(V), len(V[0])

def genIndividual():
    return np.random.rand(len(V), len(V[0]))

toolbox = base.Toolbox()
toolbox.register("attribute", genIndividual)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    V_ = individual.dot(individual.T)
    dis = V-V_
    fit = np.linalg.norm(dis)
    return fit,

def mCX(ind1_, ind2_):
    ind1, ind2 = ind1_[0], ind2_[0]
    cX_point = len(ind1)/2
    ind1[:cX_point], ind2[:cX_point] = ind2[:cX_point].copy(), ind1[:cX_point].copy()
    return ind1_, ind2_
    
def mMut(ind, mu, sigma, indpb):
    rInd = ind[0].reshape(len(ind[0])*len(ind[0][0]))
    tools.mutGaussian(rInd, mu, sigma, indpb)
    return ind

toolbox.register("mate", mCX) #tools.cxTwoPoint)
toolbox.register("mutate", mMut, mu=0, sigma=1, indpb=0.2)#tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# statistics registeration
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
#

def main():
    pop = toolbox.population(n=100)
    CXPB, MUTPB, NGEN = 0.9, 0.2, 5000

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

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

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # printing statistics
        record = stats.compile(pop)
        print "gen #%d: stats:%s"%(g, str(record['min']))
    return pop
            
pop = main()

#print pop

minInd = min(pop , key = lambda ind: ind.fitness.values[0])