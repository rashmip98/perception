from lse import least_squares_estimation
import numpy as np

def ransac_estimator(X1, X2, num_iterations=60000):
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None
    e3 = np.array([[0, 0, 1]])
    e3_hat = np.array([[0,-1,0], [1, 0, 0], [0, 0, 0]])

    for i in range(num_iterations):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]
        d = np.zeros(len(test_indices))

        E = least_squares_estimation(X1[sample_indices], X2[sample_indices])
        for i in range(np.shape(X2[test_indices])[0]):
            #print(X1[test_indices][i].shape)
            d_x2 = ((X2[test_indices][i].T@E@X1[test_indices][i])**2)/(np.linalg.norm(e3_hat@E@X1[test_indices][i])**2) #np.cross(e3,E@X1[test_indices][i].T) #np.linalg.norm(e3_hat@E@X1[test_indices][i].T)
            d_x1 = ((X1[test_indices][i].T@E.T@X2[test_indices][i])**2)/(np.linalg.norm(e3_hat@E.T@X2[test_indices][i])**2) #np.cross(e3,E@X2[test_indices][i].T)  #np.linalg.norm(e3_hat@E@X2[test_indices][i].T)
            d[i] = d_x2 + d_x1

        inliers = test_indices[np.argwhere(d<eps)]
        

        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            #print(sample_indices.flatten().shape)
            #print(inliers.flatten().shape)
            best_inliers = np.hstack((sample_indices.flatten(), inliers.flatten()))
            #print('inliers',inliers.shape)
    #print('best_inliers', best_inliers.shape)
    #print('best E', best_E.shape)
    return best_E, best_inliers