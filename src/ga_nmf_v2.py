#from deap import base, creator
import random
from deap import tools
import numpy as np
import utils
import ga_nmf_base as ga

#V = [[0,1,0,1,0],
#     [1,0,1,0,1],
#     [0,1,0,0,0],
#     [1,0,0,0,1],
#     [0,1,0,1,0]]

#V = [[0,1,0],
#     [1,0,1],
#     [0,1,0]]

#V =[[0,1,0,1,0,0,0,1],
#    [1,0,1,0,0,0,0,0],
#    [0,1,0,1,1,0,0,0],
#    [1,0,1,0,0,0,0,1],
#    [0,0,1,0,0,1,0,0],
#    [0,0,0,0,1,0,0,1],
#    [0,0,0,0,0,0,0,1],
#    [1,0,0,1,0,1,1,0]]

V = utils.read_matrix_edgeList('../resources/facebook_4039N.txt')

r_dim = 50

def genIndividual():
    return np.random.rand(len(V), r_dim)
    
def evaluate_ind(individual):
    ind_ = individual[0]
    V_ = ind_.dot(ind_.T)
    dis = V-V_
    fit = np.linalg.norm(dis)
    return fit,

def mCX_single(ind1_, ind2_):
    ind1, ind2 = ind1_[0], ind2_[0]
    cX_point = random.randint(1,len(ind1))
    ind1[:,:cX_point], ind2[:,:cX_point] = ind2[:,:cX_point].copy(), ind1[:,:cX_point].copy()
    return ind1_, ind2_

def mCX_double(ind1_, ind2_):
    ind1, ind2 = ind1_[0], ind2_[0]
    
    cX_point_1 = random.randint(0,len(ind1)-1)
    cX_point_2 = random.randint(0,len(ind1))
    
    if cX_point_1 == cX_point_2:
        cX_point_2 += 1
    elif cX_point_1 > cX_point_2:
        cX_point_1, cX_point_2 = cX_point_2, cX_point_1
        
    ind1[:,cX_point_1:cX_point_2], ind2[:,cX_point_1:cX_point_2] = ind2[:,cX_point_1:cX_point_2].copy(), ind1[:,cX_point_1:cX_point_2].copy()
    return ind1_, ind2_
    
def linear_combinaiton_CX(ind1_, ind2_):
    ind1, ind2 = ind1_[0], ind2_[0]
    rand1, rand2= random.random(), random.random()
    rand1_c, rand2_c = 1-rand1, 1-rand2
    
    ind1, ind2 = (ind1.copy()*rand1 + ind2.copy()*rand1_c), (ind1.copy()*rand2 + ind2.copy()*rand2_c)
    return ind1_, ind2_
    
def mMut(ind, mu, sigma, indpb):
    rInd = ind[0].reshape(len(ind[0])*len(ind[0][0]))
    tools.mutGaussian(rInd, mu, sigma, indpb)
    return ind
    
def additiveRule_LS(ind):
    W = ind[0]
    VW = V.dot(W)
    WWW  = W.dot(np.dot(W.T, W))
    beta = 0.1e-5
    
    for i in xrange(len(V)): # iterate over rows
        for l in xrange(r_dim): # iterate over latent dimensions
            #print i,' ',l
            new_Wij = W[i][l] + beta * (VW[i][l] - WWW[i][l])
            if new_Wij >= 0: W[i][l] = new_Wij
    return ind
    
if __name__ == '__main__':
    
    import time
    start = time.time()
    import multiprocessing
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(processes= 8,maxtasksperchild=8)
    multiprocessing.freeze_support()
    pop = ga.run_ga(pool = pool, pop_size = 50,ind_gen = genIndividual, mate = linear_combinaiton_CX, mutate = mMut, evaluate = evaluate_ind, local_search = additiveRule_LS)
    
    end = time.time()
    print(end - start)
    
    #print pop
    
    minInd = min(pop , key = lambda ind: ind.fitness.values[0])