from SwarmPackagePy import cso
import utils
import numpy as np

dataset = ('movelens 100k', '../resources/ml-100k/final_set.csv')
train, test, mSize = utils.read_data_to_train_test(dataset[1], zero_index = False)

V = utils.create_matrix(train, mSize)
maskV = np.sign(V)

r_dim = 20

def evaluate_ind(ind):
    ind_ = np.array(ind)
    ind_ = ind_.reshape((mSize[0]+mSize[1], r_dim))
    W, H = ind_[:mSize[0]], ind_[mSize[0]:]
    predV = maskV * W.dot(H.T)
    fit = utils.rmse(V, predV, len(train))#np.linalg.norm(V-predV)
    
    if np.min(ind)<0:
        fit *= 100
    return fit
   
    
if __name__ == '__main__':  
    
    method_name = 'Cuckoo_Search'
    n = 5
    iteration=10
    nest=100
    cs = cso(n = n, function = evaluate_ind, A=0, B=1, dimension = ((mSize[0]+mSize[1])*r_dim), iteration=iteration, pa=0.5, nest=nest)

    #printng results:
    minInd = cs.get_Gbest()
    W, H = minInd[:mSize[0]], minInd[mSize[0]:].T
    ga_results = utils.print_results(uMat = W, iMat = H, nFeatures = r_dim, 
                                train_data = train, test_data = test, 
                                method_name = method_name, nIterations = iteration, 
                                dataset_name = dataset[0], 
                                method_details = [('nNest',nest), ('n',n)]
                                )