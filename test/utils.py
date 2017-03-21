import numpy as np
from random import shuffle

def read_matrix_edgeList(fileName, delimiter = r' '):
    result_matrix = None
    with open(fileName, 'r') as f:
        for line in f:
            if '#' in line:
                continue
            line = [int(i) for i in line.split(delimiter)]
            if result_matrix is None:
                result_matrix = np.zeros((line[0], line[1]))
                continue
            result_matrix[line[0]][line[1]] = result_matrix[line[1]][line[0]] = 1
    return result_matrix
            
def read_edgeList(fileName, training_size = 0.7, delimiter = ' ', zero_index = True):
    edgeList = []
    with open(fileName, 'r') as f:
        line = f.readline()
        matrix_size = [int(i) for i in line.strip().split(delimiter)]
    
        if zero_index:
            for line in f:
                edgeList.append([int(i) for i in line.strip().split(delimiter)])
        else :
            for line in f:
                edgeList.append([int(i)-1 for i in line.strip().split(delimiter)])
                
        shuffle(edgeList)
    return edgeList[:int(len(edgeList) * training_size)], edgeList[int(len(edgeList) * training_size):], matrix_size

def create_matrix(edgeList, size):
    mat = np.zeros(size)
    
    for i in edgeList:
        mat[i[0]][i[1]] = mat[i[1]][i[0]] = 1
    return mat
          
if __name__ == '__main__': 
    train, test, size = read_edgeList('../resources/test_matrix')
    mat = create_matrix(train, size)
    
