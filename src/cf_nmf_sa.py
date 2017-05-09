from simanneal import Annealer
import random
import numpy as np
import utils
from scipy.special import gamma
from math import sin, pi



class CollaborativeFiltering_NMF(Annealer):

    def __init__(self, V, r_dim, _lambda=1.5, stepSize=0.01):
        self.V = V
        self.r_dim = r_dim
        self._lambda = _lambda
        self.stepSize = stepSize
        self.size = (np.sum(self.V.shape), self.r_dim)
        
        #calc sigma
        self.sigma = gamma(1+self._lambda) * sin(pi*self._lambda/2.)
        self.sigma /= self._lambda * gamma((1+self._lambda)/2.) * 2**((self._lambda-1)/2.)
        self.sigma = self.sigma**(1./self._lambda)
        ##########################
        
        self.state = self.generate_state()
        super(CollaborativeFiltering_NMF, self)#.__init__(self.state)  # important!
        
    def move(self):
        step = self.mantegna_levy_step()
        rw = self.levy_grw(step = step)
        self.state += rw
    
    def energy(self):
        W, H = self.state[:mSize[0]], self.state[mSize[0]:]
        predV = maskV * W.dot(H.T)
        rmse = utils.rmse(V, predV, len(train))
        return rmse
        
    def generate_state(self):
        return np.random.uniform(5./r_dim, 0, size = self.size)
        #r = np.random.rand(self.size)
        #r = np.random.normal(5./r_dim, .1, size = self.size)
        
    def mantegna_levy_step(self):
        u = np.random.randn(self.size[0], self.size[1])*self.sigma 
        v = np.absolute(np.random.randn(self.size[0], self.size[1]))
        step = u/np.power(v, 1./self._lambda)
        
        step = np.abs(step)
        step = np.maximum(step, 1)
        
        return step
    
    def levy_grw(self, step=None):
        levy = self._lambda * gamma(self._lambda)*sin(pi*self._lambda/2)/(pi*step**(1+self._lambda))
        return self.stepSize * levy 

        
if __name__ == '__main__':
    
    method_name = "SA"
    dataset = ('movelens 100k', '../resources/ml-100k/final_set.csv')
    #dataset = ('movelens 1m', '../resources/ml-1m/ratings.dat')
    train, test, mSize = utils.read_data_to_train_test(dataset[1], zero_index = False)
    V = utils.create_matrix(train, mSize)
    maskV = np.sign(V)
    r_dim = 50
    _lambda = 1.5
    stepSize = 1e-1
    
    cf = CollaborativeFiltering_NMF(V, r_dim, _lambda=_lambda, stepSize=stepSize)
    cf.steps = 200
    cf.updates = cf.steps/5
    cf.Tmax=.01
    cf.Tmin=.001
    
    cf.copy_strategy = "method"
    state, e = cf.anneal()
    
    print ''#cf.auto(1)
    
    W, H = state[:mSize[0]], state[mSize[0]:].T
    sa_results = utils.print_results(predMat = W.dot(H), nFeatures = r_dim, 
                                    train_data = train, test_data = test, 
                                    method_name = method_name, nIterations = cf.steps, 
                                    dataset_name = dataset[0],
                                    method_details = [
                                    ('stepSize',stepSize),
                                    ('lambda', _lambda),
                                    ('Tmax',cf.Tmax),
                                    ('Tmin',cf.Tmin)
                                    ]
                                    )