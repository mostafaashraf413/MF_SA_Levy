import numpy as np
import random
from math import sqrt

def read_data_to_train_test(fileName, delimiter = ' ', train_size = 0.8, zero_index = True):
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
    
#def test_accuracy_without_bias(test_data, matrix):
#    true_predictions = 0
#    for u,i,r in test_data:
#        if round(matrix[u][i]) == r:
#            true_predictions+=1
#    accuracy = true_predictions/float(len(test_data))
#    return accuracy
    
#def mean_squared_error_without_bias(test_data, matrix):
#    mSE = 0
#    for u,i,act in [ [t[0], t[1], t[2]] for t in test_data]:
#        pred = matrix[u][i] 
#        error = (act-pred)
#        mSE += error*error
#    return mSE/float(len(test_data))
    
def rmse(real_mat, pred_mat, n):
    return np.linalg.norm(real_mat-pred_mat) * sqrt(1./n)
    

def create_matrix(edgeList, size):
    mat = np.zeros(size)
    
    for i in edgeList:
        mat[i[0]][i[1]] = i[2]
    return mat
          
    
