from simanneal import Annealer
import random
import numpy as np
import utils
from scipy.special import gamma
from math import sin, pi



class CollaborativeFiltering_NMF(Annealer):

    def __init__(self, V, r_dim, _lambda=1.5, stepSize=0.01):
        self.V = V
        self.maskV = np.sign(V)
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
        super(CollaborativeFiltering_NMF, self)#.__init__(self.state)
        
    def move(self):
        step = self.mantegna_levy_step()
        rw = self.levy_grw(step = step)
        self.state += rw
    
    def energy(self):
        W, H = self.state[:mSize[0]], self.state[mSize[0]:]
        predV = self.maskV * W.dot(H.T)
        rmse = utils.rmse(V, predV, len(train))
        _energy = rmse
        return _energy
        
    def generate_state(self):
        return np.random.uniform((3./r_dim)**.5, (3.5/r_dim)**.5, size = self.size)
        #return np.random.normal((2.5/r_dim)**.5, 0.1, size = self.size)
        
    def mantegna_levy_step(self):
        u = np.random.randn(self.size[0], self.size[1])*self.sigma 
        v = np.absolute(np.random.randn(self.size[0], self.size[1]))
        step = u/np.power(v, 1./self._lambda)
        
        #sign = np.sign(step)
        step = np.abs(step)
        step = np.maximum(step, 1)
        #step = step*sign
        
        return step
    
    def levy_grw(self, step=None):
        levy = self._lambda * gamma(self._lambda)*sin(pi*self._lambda/2)/(pi*step**(1+self._lambda))
        return self.stepSize * levy 
        #return self.stepSize * step
        
if __name__ == '__main__':
    
    method_name = "SA"
    #dataset = ('movelens 100k', '../resources/ml-100k/final_set.csv')
    #train, test, mSize = utils.read_data_to_train_test(dataset[1], train_size=.8, zero_index = False)
    
    
    dataset = ('movelens 1m', '../resources/ml-1m/ratings.dat')
    train, _, mSize = utils.read_data_to_train_test(dataset[1]+'.tr', train_size=1., zero_index = False)
    _, test, mSize = utils.read_data_to_train_test(dataset[1]+'.ts', train_size=0.0, zero_index = False)
    
    V = utils.create_matrix(train, mSize)
    
    r_dim = 20
    _lambda = 1.5
    stepSize = 1e-2
    
    cf = CollaborativeFiltering_NMF(V, r_dim, _lambda=_lambda, stepSize=stepSize)
    cf.steps = 25
    cf.updates = cf.steps/5
    cf.Tmax=25000.0
    cf.Tmin=2.5
    
    cf.copy_strategy = "method"
    state, e = cf.anneal()
    
    print ''
    
    W, H = state[:mSize[0]], state[mSize[0]:].T
    predMat = W.dot(H)
    sa_results = utils.print_results(predMat = predMat, nFeatures = r_dim, 
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
    del predMat