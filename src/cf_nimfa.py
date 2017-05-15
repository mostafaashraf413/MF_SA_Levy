import numpy as np
import nimfa
import utils

def nimfa_factorize(V, rank=20, max_iter = 100):
    """
    Perform SNMF/R factorization on the sparse MovieLens data matrix. 
    
    Return basis and mixture matrices of the fitted factorization model.  
    :type V: `numpy.matrix`
    """
    snmf = nimfa.Snmf(V, seed="random_vcol", rank=rank, max_iter=max_iter, version='r', eta=1.,
                      beta=1e-4, i_conv=10, w_min_change=0)
    print("Algorithm: %s\nInitialization: %s\nRank: %d" % (snmf, snmf.seed, snmf.rank))
    fit = snmf()
    sparse_w, sparse_h = fit.fit.sparseness()
    print("""Stats:
            - iterations: %d
            - Euclidean distance: %5.3f
            - Sparseness basis: %5.3f, mixture: %5.3f""" % (fit.fit.n_iter,
                                                            fit.distance(metric='euclidean'),
                                                            sparse_w, sparse_h))
    return fit.basis(), fit.coef()    
    
if __name__ == '__main__':
    dataset = ('movelens 100k', '../resources/ml-100k/final_set.csv')
    #dataset = ('movelens 1m', '../resources/ml-1m/ratings.dat')
    training_data, test_data, matrix_size = utils.read_data_to_train_test(dataset[1], train_size=.8,zero_index = False)
    
    train_rating_mat = utils.create_matrix(training_data, matrix_size)
    test_rating_mat = utils.create_matrix(test_data, matrix_size)
    
    avg_rating = np.mean(np.array(training_data)[:,2])
    for i in xrange(train_rating_mat.shape[0]):
        for j in xrange(train_rating_mat.shape[1]):
            if train_rating_mat[i][j] == 0:
                 train_rating_mat[i][j] = avg_rating
    
    rank = 20
    max_iter = 5
    W, H = nimfa_factorize(train_rating_mat, rank=rank, max_iter=max_iter)
    
    utils.print_results(predMat = np.array(W.dot(H)), nFeatures = rank,
                            train_data = training_data, test_data = test_data,
                            method_name = 'NIMFA', nIterations = max_iter, dataset_name = dataset[0])
    
    