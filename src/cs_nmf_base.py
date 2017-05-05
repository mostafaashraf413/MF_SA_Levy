from deap import base, creator
import random
from deap import tools
import numpy as np
import matplotlib.pyplot as plt

class CS_NMF:

    def __init__(self):
        # statistics registeration
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        #stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        self.toolbox = base.Toolbox()
    
    def __clone(self, _ind):
        ind = self.toolbox.nest(n=1)[0]
        ind[:] = _ind.copy()
        return ind
        
    def step_decay_factor(self, _max, _min, nIter):
        return float(_max-_min)/nIter
        
    def rep_cuckoo(self, cuckoo):
        r_dim = cuckoo.shape[1]
        cuckoo[:] = self.toolbox.nest(n=1)[0][:] #np.random.normal(5./r_dim, .1, size = cuckoo.shape)#
        cuckoo.fitness.values = self.toolbox.evaluate(cuckoo)
        return cuckoo
    
    def run_cs(self, pa = 0.25, nIter = 100, ind_type = np.ndarray, ind_size = None, nCuckoos = 50, ind_gen = None,
                l_rw = None, g_rw = None, select = tools.selRandom, evaluate = None, stepFunction = None, _lambda = 1.5,
                max_stepSize = 0.1, min_stepSize = 0.001,curve_label = "CS"):
                
        df = self.step_decay_factor(max_stepSize, min_stepSize, nIter)
        stepSize = max_stepSize
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Cuckoo", ind_type, fitness=creator.FitnessMin)
        
        self.toolbox.register("attribute", ind_gen)
        self.toolbox.register("cuckoo", tools.initRepeat, creator.Cuckoo, self.toolbox.attribute, n=ind_size)
        self.toolbox.register("nest", tools.initRepeat, list, self.toolbox.cuckoo)
        
        cuckoos = self.toolbox.nest(n=nCuckoos)
        
        self.toolbox.register("stepFunction", stepFunction, _lambda, cuckoos[0].shape)
        self.toolbox.register("l_ranWalk", l_rw, _lambda, pa)
        self.toolbox.register("g_ranWalk", g_rw, _lambda,)
        self.toolbox.register("select", select)
        self.toolbox.register("evaluate", evaluate)
        
        # Evaluate the entire nest
        fitnesses = map(self.toolbox.evaluate, cuckoos)
        for cuckoo, fit in zip(cuckoos, fitnesses):
            cuckoo.fitness.values = fit
        
        min_fit_lst = []
        for g in range(nIter):
            
            # global random walk:
            cuckoo = self.toolbox.select(cuckoos, 1)[0]
            step = self.toolbox.stepFunction()
            cuckoo = self.toolbox.g_ranWalk(stepSize, self.__clone(cuckoo), step)
            cuckoo.fitness.values = self.toolbox.evaluate(cuckoo)
            
            ri = random.randint(0, nCuckoos-1)
            cuckoos[ri] = cuckoo if cuckoo.fitness.values[0] < cuckoos[ri].fitness.values[0] else cuckoos[ri]
            
            #replace worst cuckoos
            worst_cuckoos = tools.selWorst(cuckoos, int(pa*len(cuckoos)))
            map(self.rep_cuckoo, worst_cuckoos)
            
            #local random walk:
            best_cuckoo = tools.selBest(cuckoos, 1)[0]
            new_cuckoos = map(lambda c: self.toolbox.l_ranWalk(stepSize, self.__clone(c), best_cuckoo, self.toolbox.stepFunction()), cuckoos)
            fitnesses = map(self.toolbox.evaluate, new_cuckoos)  
            for i in xrange(len(new_cuckoos)):
                cuckoos[i][:] = new_cuckoos[i].copy()
                cuckoos[i].fitness.values = fitnesses[i]
             
            stepSize = stepSize - df
            
            # printing statistics
            record = self.stats.compile(cuckoos)
            min_fit = record['min']
            max_fit = record['max']
            std = record['std']
            min_fit_lst.append(min_fit)
            print "gen #%d: stats min:%f max:%f std:%f"%(g, min_fit, max_fit, std)
        plt.plot(min_fit_lst, label=curve_label)
        plt.legend()
        plt.show()
        return cuckoos
        
        
        
        
        
        
        