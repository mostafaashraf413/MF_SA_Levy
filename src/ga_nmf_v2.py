#from deap import base, creator
import random
from deap import tools
import numpy as np
import utils
import ga_nmf_base as ga
from scipy.stats import levy_stable as levy
from scipy.special import gamma
from math import sin, pi

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

r_dim = 50

def generate_ind():
    return np.random.rand(r_dim)
      
def evaluate_ind(ind):
    V_ = ind.dot(ind.T)
    dis = V-V_
    fit = np.linalg.norm(dis)
    return fit,

def mCX_single(ind1, ind2):
    cX_point = random.randint(1,len(ind1))
    ind1[:,:cX_point], ind2[:,:cX_point] = ind2[:,:cX_point].copy(), ind1[:,:cX_point].copy()
    return ind1, ind2

def mCX_double(ind1, ind2):
    cX_point_1 = random.randint(0,len(ind1)-1)
    cX_point_2 = random.randint(0,len(ind1))
    
    if cX_point_1 == cX_point_2:
        cX_point_2 += 1
    elif cX_point_1 > cX_point_2:
        cX_point_1, cX_point_2 = cX_point_2, cX_point_1
        
    ind1[:,cX_point_1:cX_point_2], ind2[:,cX_point_1:cX_point_2] = ind2[:,cX_point_1:cX_point_2].copy(), ind1[:,cX_point_1:cX_point_2].copy()
    return ind1, ind2
    
def linear_combinaiton_CX(ind1, ind2):
    rand1, rand2= random.random(), random.random()
    rand1_c, rand2_c = 1-rand1, 1-rand2
    
    ind1[:], ind2[:] = (ind1.copy()*rand1 + ind2.copy()*rand1_c), (ind1.copy()*rand2 + ind2.copy()*rand2_c)
    return ind1, ind2
    
def mMut(ind, indpb):
    mu=0
    sigma=1
    tools.mutGaussian(ind, mu, sigma, indpb)
    return ind
    
def levyMut(ind, indpb):
    ind += 0.1 * levy.rvs(alpha = 1.5, beta=0.5, size=(len(ind), len(ind[0])))
    return ind
     
def mantegna_levy_step(beta=1.5):
    sigma = gamma(1+beta) * sin(pi*beta/2.)
    sigma /= ( beta * gamma((1+beta)/2) * pow(2, (beta-1)/2.) )
    sigma = pow(sigma , 1./beta)
    
    u = random.normal(scale=sigma)
    v = abs(random.normal())
    
    step = u/pow(v, 1./beta)
    
    return step
    
def additiveRule_LS(ind):

    W = ind
    VW = V.dot(W)
    WWW  = W.dot(np.dot(W.T, W))
    beta = 0.1e-5
    
    W += beta * (VW - WWW)
    return ind
    
def multiplicativeRule_LS(ind):
    W = ind
    VW = V.dot(W)
    WWW  = W.dot(np.dot(W.T, W))
    beta = 0.5
    
    W *= (1 - beta + (beta * VW / WWW))
    return ind
    
def gradient_descent_LS(ind):
    W = ind
    n_points = 1000
    point_lst = zip(np.random.randint(low=0, high=len(V), size=n_points), np.random.randint(low=0, high=len(V[0]), size=n_points))
    lr = 0.02
    for i,j in point_lst:
        e = V[i][j] - np.dot(W[i], W[j].T)
        W[i] += lr*e*W[j]
        W[j] += lr*e*W[i] 
    return ind 

    
if __name__ == '__main__':
  
    pop = ga.run_ga(ind_size = len(V), pop_size = 50, mate = mCX_double, mutate = levyMut, MUTPB = 0.4, 
                    evaluate = evaluate_ind, local_search = multiplicativeRule_LS, CXPB = 0.9, LSPB = 0.3,
                    ind_gen = generate_ind)
   
    #print pop
    
    minInd = min(pop , key = lambda ind: ind.fitness.values[0])