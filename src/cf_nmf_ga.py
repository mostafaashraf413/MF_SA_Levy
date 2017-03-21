#from deap import base, creator
import random
from deap import tools
import numpy as np
import utils
import ga_nmf_base as ga
from scipy.stats import levy_stable as levy
from scipy.special import gamma
from math import sin, pi


train, test, mSize = utils.read_data_to_train_test('../resources/ml-100k/final_set.csv', zero_index = False)

V = utils.create_matrix(train, mSize)
maskV = np.sign(V)

r_dim = 20
eps = 1e-5

def generate_ind():
    r = np.random.rand(r_dim)
    r = np.maximum(r, eps)
    return r
      
def evaluate_ind(ind):
    W, H = ind[:mSize[0]], ind[mSize[0]:]
    predV = maskV * W.dot(H.T)
    norm = np.linalg.norm(V-predV)
    return norm,
    
def mCX_single(ind1, ind2):
    cX_point = random.randint(1,len(ind1))
    ind1[:,:cX_point], ind2[:,:cX_point] = ind2[:,:cX_point].copy(), ind1[:,:cX_point].copy()
    return ind1, ind2

def mCX_double_vertically(ind1, ind2):
    cX_point_1 = random.randint(0,len(ind1[0])-1)
    cX_point_2 = random.randint(0,len(ind1[0]))
    
    if cX_point_1 == cX_point_2:
        cX_point_2 += 1
    elif cX_point_1 > cX_point_2:
        cX_point_1, cX_point_2 = cX_point_2, cX_point_1
        
    ind1[:,cX_point_1:cX_point_2], ind2[:,cX_point_1:cX_point_2] = ind2[:,cX_point_1:cX_point_2].copy(), ind1[:,cX_point_1:cX_point_2].copy()
    return ind1, ind2
    
def mCX_double_horizontally(ind1, ind2):
    cX_point_1 = random.randint(0,len(ind1)-1)
    cX_point_2 = random.randint(0,len(ind1))
    
    if cX_point_1 == cX_point_2:
        cX_point_2 += 1
    elif cX_point_1 > cX_point_2:
        cX_point_1, cX_point_2 = cX_point_2, cX_point_1
        
    ind1[cX_point_1:cX_point_2], ind2[cX_point_1:cX_point_2] = ind2[cX_point_1:cX_point_2].copy(), ind1[cX_point_1:cX_point_2].copy()
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
    ind = np.maximum(ind, eps)
    return ind
    
def levyMut(ind, indpb):
    ind += 0.1 * levy.rvs(alpha = 1.5, beta=0.5, size=(len(ind), len(ind[0])))
    ind = np.maximum(ind, eps)
    return ind
     
#def mantegna_levy_step(beta=1.5):
#    sigma = gamma(1+beta) * sin(pi*beta/2.)
#    sigma /= ( beta * gamma((1+beta)/2) * pow(2, (beta-1)/2.) )
#    sigma = pow(sigma , 1./beta)
#    
#    u = random.normal(scale=sigma)
#    v = abs(random.normal())
#    
#    step = u/pow(v, 1./beta)
#    
#    return step

def least_square_LS(ind):
    #ind[:mSize[0]] = np.linalg.lstsq(ind[mSize[0]:].T, V)[0]
    ind[mSize[0]:] = np.linalg.lstsq(ind[:mSize[0]], V)[0].T
    return ind

def multiplicativeRule_LS(ind):
    #W = ind
    #VW = V.dot(W)
    #WWW  = W.dot(np.dot(W.T, W))
    #beta = 0.5
    #
    #W *= (1 - beta + (beta * VW / WWW))
    #return ind
    pass
    
def gradient_descent_LS(ind):
    #W = ind
    #n_points = 1000
    #point_lst = zip(np.random.randint(low=0, high=len(V), size=n_points), np.random.randint(low=0, high=len(V[0]), size=n_points))
    #lr = 0.02
    #for i,j in point_lst:
    #    e = V[i][j] - np.dot(W[i], W[j].T)
    #    W[i] += lr*e*W[j]
    #    W[j] += lr*e*W[i] 
    #return ind 
    pass
    
if __name__ == '__main__':
  
    pop = ga.run_ga(ind_size = mSize[0]+mSize[1], pop_size = 70, mate = mCX_double_horizontally, mutate = levyMut, MUTPB = 0.2, 
                    evaluate = evaluate_ind, local_search = least_square_LS, CXPB = 0.9, LSPB = 0.0,
                    ind_gen = generate_ind, new_inds_ratio = 0.3)
   
    #print pop
    
    minInd = min(pop , key = lambda ind: ind.fitness.values[0])