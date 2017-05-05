import numpy as np
import random
from math import sqrt

def read_data_to_train_test(fileName, delimiter = ' ', train_size = 0.9, zero_index = True):
    test_data = None
    tmp_lst = []
    matrix_size = None
    
    with open(fileName, 'r') as f:
        line = f.readline().strip().split(delimiter)
        matrix_size = (int(line[0]), int(line[1]))
        if zero_index:
            for line in f:
                line = line.strip().split(delimiter) 
                tmp_lst.append( [int(line[0]), int(line[1]), float(line[2])] )
        else:
            for line in f:
                line = line.strip().split(delimiter) 
                tmp_lst.append( [int(line[0])-1, int(line[1])-1, float(line[2])] )
            
    random.shuffle(tmp_lst)
    cut_point = int(len(tmp_lst)*train_size)
    test_data, training_data = tmp_lst[cut_point:], tmp_lst[:cut_point]
    
    return training_data, test_data, matrix_size
    
def rmse(real_mat, pred_mat, n):
    return np.linalg.norm(real_mat-pred_mat) * sqrt(1./n)
    

def create_matrix(edgeList, size):
    mat = np.zeros(size)
    
    for i in edgeList:
        mat[i[0]][i[1]] = i[2]
    return mat
          
def print_results(uMat=None, iMat=None, predMat = None, nFeatures=None, train_data=None, test_data=None, method_name=None, 
                    nIterations=None, dataset_name=None, method_details=[]):
    results ='\n############# results of %s method: \n'%(method_name) 
    results += '## dataset: %s \n'%(dataset_name)
    results += '## number of latent features: %d \n'%(nFeatures)
    results += '## number of training iteratins: %d \n'%(nIterations)
    
    for d in method_details:
        results += '## %s: %s \n'%(str(d[0]), str(d[1]))
    
    if uMat and iMat:
        predMat = uMat.dot(iMat)
    
    trainMat = create_matrix(train_data, predMat.shape)
    testMat = create_matrix(test_data, predMat.shape)
    
    #calculating rmse
    train_rmse = rmse(trainMat, predMat*np.sign(trainMat), len(train_data))
    test_rmse = rmse(testMat, predMat*np.sign(testMat), len(test_data))
    
    results += '## training RMSE: %f \n'%(train_rmse)
    results += '## testing RMSE: %f \n'%(test_rmse)
    
    results +='########################################################'
    
    with open('../results.txt', 'a') as f:
        f.writelines(results)
    
    print results
    return results
    
    
    
