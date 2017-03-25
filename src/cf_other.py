#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import numpy as np
import utils
import matplotlib.pyplot as plt


#def mean_squared_error(test_data, matrix, biasU, biasI, biasG):
#    mSE = 0
#    for u,i,act in [ [t[0], t[1], t[2]] for t in test_data]:
#        pred = matrix[u][i] + biasU[u] + biasI[i] + biasG
#        error = (act-pred)
#        mSE += error*error
#    return mSE/float(len(test_data))

#def mean_squared_error_without_bias(test_data, matrix):
#    mSE = 0
#    for u,i,act in [ [t[0], t[1], t[2]] for t in test_data]:
#        pred = matrix[u][i] 
#        error = (act-pred)
#        mSE += error*error
#    return mSE/float(len(test_data))
    

    
####################################################################################
#"Luo, Xin, et al. "An efficient non-negative matrix-factorization-based approach to 
# collaborative filtering for recommender systems." IEEE Transactions on Industrial 
# Informatics 10.2 (2014): 1273-1284."‏
def collaborative_filtering_rsnmf(training_data = None, rating_matrix_size = None,
                                    nLatent_features=None, rf=0.06, nIterations=100):
    W = np.random.rand(rating_matrix_size[0], nLatent_features) 
    H = np.random.rand(nLatent_features, rating_matrix_size[1])
    Wup = np.zeros((rating_matrix_size[0], nLatent_features)) 
    Wdown = np.zeros((rating_matrix_size[0], nLatent_features)) 
    Hup = np.zeros((nLatent_features, rating_matrix_size[1]))
    Hdown = np.zeros((nLatent_features, rating_matrix_size[1]))
    
    for t in xrange(nIterations):
        for u,i,r in training_data:
            r_ = W[u].dot(H[:,i])
            for k in xrange(nLatent_features):
                Wup[u][k] += H[k][i]*r
                Wdown[u][k] += H[k][i]*r_
                
        for u in xrange(rating_matrix_size[0]):
            for k in xrange(nLatent_features):
                Wdown[u][k] += rf*W[u][k]
                W[u][k] *= Wup[u][k]/Wdown[u][k]
                
        for u,i,r in training_data:
            r_ = W[u].dot(H[:,i])
            for k in xrange(nLatent_features):
                Hup[k][i] += W[u][k]*r
                Hdown[k][i] += W[u][k]*r_
                
        for i in xrange(rating_matrix_size[1]):
            for k in xrange(nLatent_features):
                Hdown[k][i] += rf*H[k][i]
                H[k][i] *= Hup[k][i]/Hdown[k][i]
                
        Wup[:], Wdown[:], Hup[:], Hdown[:] = 0,0,0,0
        
        rating_matrix = W.dot(H)
        mse = mean_squared_error_without_bias(training_data, rating_matrix)
        print 'iteration #%d: accuracy = %f, mse = %f'%(t+1, mse)
        
    return W, H


###################################################################################
#"Zhang, Sheng, et al. "Learning from incomplete ratings using non-negative matrix 
# factorization." Proceedings of the 2006 SIAM International Conference on Data Mining. 
# Society for Industrial and Applied Mathematics, 2006."
########################################
# http://stackoverflow.com/questions/22767695/python-non-negative-matrix-factorization-that-handles-both-zeros-and-missing-dat
def collaborative_filtering_wnmf(training_data = None, rating_matrix_size = None, 
                                               nLatent_features=None, beta = 0.5, nIterations=100): 
    eps = 1e-5                            
    V = utils.create_matrix(training_data, rating_matrix_size)
    W = np.random.rand(rating_matrix_size[0], nLatent_features)
    W = np.maximum(W, eps)
    H = np.random.rand(nLatent_features, rating_matrix_size[1])#np.linalg.lstsq(W,V)[0]
    H = np.maximum(H, eps)
    NG = np.sign(V)
    NG_V = NG*V
    
    rmse_lst = []
    
    for i in xrange(nIterations):
        VH = np.dot(NG_V, H.T)
        WHH = np.dot(NG*W.dot(H), H.T)+eps
        W = W *(1-beta + (beta*(VH/WHH)))
        W = np.maximum(W, eps)
        
        WV = np.dot(W.T, NG_V)
        WWH = np.dot(W.T, NG*W.dot(H))+eps
        H = H *(1-beta + (beta*(WV/WWH)))
        H = np.maximum(H, eps)
        
        del WV, WWH, VH, WHH
        
        rating_matrix = W.dot(H)
        
        rmse = utils.rmse(V, rating_matrix*NG, len(training_data))
        print 'iteration #%d: rmse = %f'%(i+1, rmse)
        rmse_lst.append(rmse)
        
    plt.plot(rmse_lst, label="WNMF")
    plt.legend()
    plt.show()
    return W, H



########################################################################################
#"Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix factorization techniques 
# for recommender systems." Computer 42.8 (2009)."‏
def collaborative_filtering_SGD(training_data = None, rating_matrix_size = None, 
                                  nLatent_features=None, rf=0.01, lr=0.01, nIterations=100):
                            
    training_data = np.array(training_data)
    
    rMat = np.random.normal(scale=1./nLatent_features, size = (rating_matrix_size[0], nLatent_features))
    lMat = np.random.normal(scale=1./nLatent_features, size = (rating_matrix_size[1], nLatent_features))
    #rMat = np.random.rand(rating_matrix_size[0], nLatent_features)
    #lMat = np.random.rand(rating_matrix_size[1], nLatent_features)
    
    biasU = np.zeros(rating_matrix_size[0])
    biasI = np.zeros(rating_matrix_size[1])
    #biasU = np.random.rand(rating_matrix_size[0]) 
    #biasI = np.random.rand(rating_matrix_size[1])
    
    biasG = np.mean(training_data[:,2])
    
    real_mat = utils.create_matrix(training_data, rating_matrix_size)
    sm = np.sign(real_mat)
    rmse_lst = []
    
    for k in xrange(nIterations):
        iterError = 0
        
        for u,i,y in [[td[0], td[1], td[2]] for td in  training_data]:
                                                  
            e = y - biasG - biasU[u] - biasI[i] - np.dot(rMat[u], lMat[i].T)
            
            biasU[u] += lr*(e - (rf*biasU[u]))
            biasI[i] += lr*(e - (rf*biasI[i]))
            rMat[u] += lr*(e * lMat[i] - (rf*rMat[u]))
            lMat[i] += lr*(e * rMat[u] - (rf*lMat[i]))
            
            iterError += abs(e)
        
        #####pritning progress
        biasU_mat = np.repeat(biasU, len(real_mat[0])).reshape((len(real_mat), len(real_mat[0])))
        biasI_mat = np.repeat(biasI, len(real_mat)).reshape((len(real_mat[0]) , len(real_mat))).T
        pred_mat = rMat.dot(lMat.T) + biasU_mat + biasI_mat + biasG
        
        rmse = utils.rmse(real_mat, pred_mat*sm, len(training_data))
        print "Iteration %d, training error= %f, rmse= %f"%(k, iterError, rmse)
        rmse_lst.append(rmse)
                
    biasU_mat = np.repeat(biasU, len(real_mat[0])).reshape((len(real_mat), len(real_mat[0])))
    biasI_mat = np.repeat(biasI, len(real_mat)).reshape((len(real_mat[0]) , len(real_mat))).T
    pred_mat = rMat.dot(lMat.T) + biasU_mat + biasI_mat + biasG
    print 'trainig rmse = ', utils.rmse(real_mat, pred_mat*sm, len(training_data))
    
    plt.plot(rmse_lst, label="SGD")
    plt.legend()
    plt.show()
    return rMat, lMat, biasU, biasI, biasG



if __name__ == '__main__':

    training_data, test_data, matrix_size = utils.read_data_to_train_test('../resources/ml-100k/final_set.csv', zero_index = False)
    
    #training_data, tmp, matrix_size = read_data_to_train_test("coll_filtering_datasets/ml-100k/ua.base", 
    #                                    train_size = 1, zero_index=False)
    #                                    
    #tmp, test_data, matrix_size = read_data_to_train_test("coll_filtering_datasets/ml-100k/ua.test", 
    #                                    train_size = 0, zero_index=False)                                  
    #del tmp
    
    test_rating_mat = utils.create_matrix(test_data, matrix_size)
    training_algorithm = 2
    
    #for SGD
    if training_algorithm == 1:
        rMat, lMat, biasU, biasI, biasG = collaborative_filtering_SGD(training_data = training_data, 
            rating_matrix_size = matrix_size, nLatent_features=20, rf=0.02, lr=0.01, nIterations=100)
        
        biasU_mat = np.repeat(biasU, len(test_rating_mat[0])).reshape((len(test_rating_mat), len(test_rating_mat[0])))
        biasI_mat = np.repeat(biasI, len(test_rating_mat)).reshape((len(test_rating_mat[0]) , len(test_rating_mat))).T
        
        rating_matrix = rMat.dot(lMat.T) + biasU_mat + biasI_mat + biasG
        print "testing rmse = ", utils.rmse(test_rating_mat, rating_matrix*np.sign(test_rating_mat), len(test_data))#mean_squared_error(test_data, rating_matrix, biasU, biasI, biasG)
    
    #for WMNF
    elif training_algorithm == 2:
        #training_data = map(lambda x: [x[0],x[1],x[2]/5.0] , training_data)
        #test_data = map(lambda x: [x[0],x[1],x[2]/5.0] , test_data)
        W, H = collaborative_filtering_wnmf(training_data = training_data, beta = 1,
                            rating_matrix_size = matrix_size, nLatent_features=20, nIterations=100)
        rating_matrix = W.dot(H)
        print "testing rmse = ",utils.rmse(test_rating_mat , rating_matrix*np.sign(test_rating_mat), len(test_data))
    
    #for RSNMF        
    elif training_algorithm == 3:
        W, H = collaborative_filtering_rsnmf(training_data = training_data, rf=0.15,
                rating_matrix_size = matrix_size, nLatent_features=50, nIterations=100)
                
        rating_matrix = W.dot(H)
        print "testing MSE = ",mean_squared_error_without_bias(test_data, rating_matrix)
