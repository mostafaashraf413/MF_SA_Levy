import numpy as np

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
            
        
if __name__ == '__main__': 
    data = read_matrix_edgeList('../resources/test_matrix')
    
