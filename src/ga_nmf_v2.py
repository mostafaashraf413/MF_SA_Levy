#from deap import base, creator
import random
from deap import tools
import numpy as np
import utils
import ga_nmf_base as ga
from scipy.stats import levy_stable as levy

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
V = np.array(V)

r_dim = 5

def genIndividual():
    return np.random.rand(len(V), r_dim)
    
def evaluate_ind(individual):
    ind_ = individual[0]
    V_ = ind_.dot(ind_.T)
    dis = V-V_
    fit = np.linalg.norm(dis)
    return fit,

def mCX_single(ind1_, ind2_):
    ind1_, ind2_ = ind1_.copy(), ind2_.copy()
    ind1, ind2 = ind1_[0], ind2_[0]
    cX_point = random.randint(1,len(ind1))
    ind1[:,:cX_point], ind2[:,:cX_point] = ind2[:,:cX_point].copy(), ind1[:,:cX_point].copy()
    return ind1_, ind2_

def mCX_double(ind1_, ind2_):
    ind1_, ind2_ = ind1_.copy(), ind2_.copy()
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
    ind1_, ind2_ = ind1_.copy(), ind2_.copy()
    ind1, ind2 = ind1_[0], ind2_[0]
    rand1, rand2= random.random(), random.random()
    rand1_c, rand2_c = 1-rand1, 1-rand2
    
    ind1, ind2 = (ind1.copy()*rand1 + ind2.copy()*rand1_c), (ind1.copy()*rand2 + ind2.copy()*rand2_c)
    return ind1_, ind2_
    
def mMut(ind, indpb):
    mu=0
    sigma=1
    rInd = ind[0].reshape(len(ind[0])*len(ind[0][0]))
    ind = tools.mutGaussian(rInd, mu, sigma, indpb)
    return ind
    
def levyMut(ind_, indpb):
    if random.random() > indpb:
        return ind_
    ind_ = ind_.copy()
    ind = ind_[0]
    
    ind += 0.01 * levy.rvs(alpha = 1.5, beta=0.5, size=(len(ind), len(ind[0])))
    
    return ind_
     
    
def additiveRule_LS(ind):

    W = ind[0]
    VW = V.dot(W)
    WWW  = W.dot(np.dot(W.T, W))
    beta = 0.1e-5
    
    W += beta * (VW - WWW)
    return ind
    
def multiplicativeRule_LS(ind):
    W = ind[0]
    VW = V.dot(W)
    WWW  = W.dot(np.dot(W.T, W))
    beta = 0.5
    
    W *= (1 - beta + (beta * VW / WWW))
    return ind
    
def gradient_descent_LS(ind):
    #fb = evaluate_ind(ind)
    
    W = ind[0]
    E = V - W.dot(W.T)
    lr = 0.1e-2
    rf = 1
    
    for i in xrange(len(E)): 
        for j in xrange(len(E[0])):
            print i," ",j
            W[i] = W[i].copy() + lr*(E[i][j]*W[j].copy() - rf*W[i].copy())
            W[j] = W[j].copy() + lr*(E[i][j]*W[i].copy() - rf*W[j].copy())
            
    #fa = evaluate_ind(ind)
    #print 'fitness before LS = %d, fitness after = %d'%(fb[0],fa[0])
    return ind 
    
    
if __name__ == '__main__':
  
    pop = ga.run_ga(pop_size = 50, ind_gen = genIndividual, mate = linear_combinaiton_CX, 
                    mutate = levyMut, MUTPB = 0.2, evaluate = evaluate_ind, local_search = multiplicativeRule_LS)
   
    #print pop
    
    minInd = min(pop , key = lambda ind: ind.fitness.values[0])