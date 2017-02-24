# -*- coding: utf-8 -*-
import numpy as np
import logging

logger = logging.getLogger(__file__.split('/')[-1])
logger.setLevel(logging.INFO)

def frobeniusNorm(mat1, mat2):
    dist = 0
    for i in xrange(len(mat1)):
        for j in xrange(len(mat1[0])):
            dist += pow(mat1[i][j] - mat2[i][j], 2)
    dist = np.sqrt(dist)
    return dist

def stopLearning(minError, currentError):
    if currentError <= minError:
        return True
    else: return False

"""
-missing elements in the matrix must be zeros.
-the update rule for this function is taken from:
    - Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization." Advances in neural information processing systems. 2001.‏APA	
    - Duan, Liang, et al. "Scaling up Link Prediction with Ensembles." Proceedings of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 2016.‏
"""
#def nmf_multiplicative(origMat, nLatentFeatures = 10, nSteps = 50, beta = 0.005, minError = 0.02):
#    
#    leftMat = np.random.rand(len(origMat), nLatentFeatures)
#    rightMat = np.random.rand(nLatentFeatures, len(origMat[0]))
#    
#    logger.info('factorize_MEDMR has started:')
#    for step in xrange(nSteps):
#        
#        ##### for left matrix
#        oDotR = origMat.dot(rightMat.T)
#        lDotRTrans  = leftMat.dot(np.dot(rightMat, rightMat.T))
#        
#        ##### for right matrix
#        lDotO = np.dot( leftMat.T ,origMat)
#        lTransDotR = leftMat.T.dot(np.dot(leftMat, rightMat))
#
#        for i in xrange(len(origMat)): # iterate over rows
#            for j in xrange(len(origMat[0])): # iterate over columns
#                
#                for l in xrange(nLatentFeatures): # iterate over latent dimensions
#                    leftMat[i][l] = leftMat[i][l] * (1 - beta + (beta * oDotR[i][l] / lDotRTrans[i][l]))
#                    rightMat[l][j] = rightMat[l][j] * (1 - beta + (beta * lDotO[l][j] / lTransDotR[l][j]))
#         
#        #calculate the distance
#        dist = validation.sqrEuclideanDistance(origMat, leftMat.dot(rightMat))
#        logger.info('iteration #%d, error distane = %d'%(step, dist))
#        #calculate the error to stop learning 
#        if stopLearning(dist):
#            break
#    logger.info('factorization is done!')
#        
#    return leftMat, rightMat
        
###################################################### 

"""
-missing elements in the matrix must be zeros.
-the update rule for this function is taken from:
    - Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization." Advances in neural information processing systems. 2001.‏APA	
    - Duan, Liang, et al. "Scaling up Link Prediction with Ensembles." Proceedings of the Ninth ACM International Conference on Web Search and Data Mining. ACM, 2016.‏
"""
#def nmf_multiplicative_sym(origMat, nLatentFeatures = 10, nSteps = 50, beta = 0.5, minError = 1):
#    
#    fMat = np.random.rand(len(origMat), nLatentFeatures)
#    logger.info('factorize_sym_MEDMR has started:')
#    for step in xrange(nSteps):
#        
#        oDotF = origMat.dot(fMat)
#        fDotFF  = fMat.dot(np.dot(fMat.T, fMat))
#
#        for i in xrange(len(origMat)): # iterate over rows
#                for l in xrange(nLatentFeatures): # iterate over latent dimensions
#                    fMat[i][l] = fMat[i][l] * (1 - beta + (beta * oDotF[i][l] / fDotFF[i][l]))
#        
#        #calculate the distance
#        dist = validation.sqrEuclideanDistance(origMat, fMat.dot(fMat.T))
#        logger.info('iteration #%d, error distane = %d'%(step, dist))                                       
#        #calculate the error to stop learning 
#        if stopLearning(dist):
#            break
#    logger.info('factorization is done!')
#        
#    return fMat
#        
###################################################### 

def nmf_additive(V, nLatentFeatures = 10, nSteps = 50, beta = 0.005, minError = 0.02):
    
    V = np.array(V)
    W = np.random.rand(len(V), nLatentFeatures)
    H = np.random.rand(nLatentFeatures, len(V[0]))
    
    logger.info('nmf_additive has started:')
    for step in xrange(nSteps):
        
        ##### for left matrix (W)
        VH = V.dot(H.T)
        WHH  = W.dot(np.dot(H, H.T))
        
        ##### for right matrix (H)
        WV = np.dot( W.T ,V)
        WWH = W.T.dot(np.dot(W, H))

        for i in xrange(len(V)): # iterate over rows
            for j in xrange(len(V[0])): # iterate over columns  
                for l in xrange(nLatentFeatures): # iterate over latent dimensions
                    new_Wij = W[i][l] + beta * (VH[i][l] - WHH[i][l])
                    new_Hij = H[l][j] + beta * (WV[l][j] - WWH[l][j])
                    if new_Wij >= 0: W[i][l] = new_Wij  
                    if new_Hij >= 0: H[l][j] = new_Hij
         
        #calculate the distance
        dist = frobeniusNorm(V, W.dot(H))
        logger.info('iteration #%f, error distane = %f'%(step, dist))
        #calculate the error to stop learning 
        if stopLearning(minError, dist):
            break
    logger.info('factorization is done!')
        
    return W, H
        
###################################################### 

def nmf_additive_sym(V, nLatentFeatures = 10, nSteps = 50, beta = 0.005, minError = 0.02):
    
    V = np.array(V)
    W = np.random.rand(len(V), nLatentFeatures)
    
    logger.info('nmf_additive symmetric has started:')
    for step in xrange(nSteps):
        
        ##### for left matrix (W)
        VW = V.dot(W)
        WWW  = W.dot(np.dot(W.T, W))
    
        for i in xrange(len(V)): # iterate over rows
            for j in xrange(len(V[0])): # iterate over columns  
                for l in xrange(nLatentFeatures): # iterate over latent dimensions
                    new_Wij = W[i][l] + beta * (VW[i][l] - WWW[i][l])
                    if new_Wij >= 0: W[i][l] = new_Wij
         
        #calculate the distance
        dist = frobeniusNorm(V, W.dot(W.T))
        logger.info('iteration #%f, error distane = %f'%(step, dist))
        #calculate the error to stop learning 
        if stopLearning(minError, dist):
            break
    logger.info('factorization is done!')
        
    return W
        
###################################################### 
          
# testing  
if __name__ == "__main__":
    #mat =[[0,1,0,1,0,0,0,1],
    #     [1,0,1,0,0,0,0,0],
    #     [0,1,0,1,1,0,0,0],
    #     [1,0,1,0,0,0,0,1],
    #     [0,0,1,0,0,1,0,0],
    #     [0,0,0,0,1,0,0,1],
    #     [0,0,0,0,0,0,0,1],
    #     [1,0,0,1,0,1,1,0]]
    mat = np.random.randint(0, 2, (50, 50))
    mat = (mat+mat.T)/2
    
    #w,h = nmf_additive(mat, 5, 1000, 0.001)
    w = nmf_additive_sym(mat, 40, 1000, 0.001)