import numpy as np
def epipole(u,v,smin,thresh,num_iterations = 1000):
    ''' Takes flow (u,v) with confidence smin and finds the epipole using only the points with confidence above the threshold  
        (for both sampling and finding inliers)
        u, v and smin are (w,h), thresh is a scalar
        output should be best_ep and inliers, which have shapes, respectively (3,) and (n,) 
    '''


    arg_thresh = np.where(smin.flatten()>thresh)
    xp, yp = np.meshgrid(np.linspace(-u.shape[0]/2, u.shape[0]/2 -1, u.shape[0]), np.linspace(-u.shape[1]/2 , u.shape[1]/2 -1, u.shape[1]))
    u_full = np.hstack((u.flatten().reshape(-1,1), v.flatten().reshape(-1,1), np.zeros(u.flatten().shape[0]).reshape(-1,1)))
    x_full = np.hstack((xp.flatten().reshape(-1,1), yp.flatten().reshape(-1,1), np.ones(u.flatten().shape[0]).reshape(-1,1) ))

    sample_size = 2

    eps = 10**-2

    best_num_inliers = -1
    best_inliers = None
    best_ep = None
    
    for i in range(num_iterations):
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(0,np.sum((smin>thresh))))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]
        
        u_sample = u_full[arg_thresh[0][sample_indices], :]
        x_sample = x_full[arg_thresh[0][sample_indices], :]
        
        A = np.cross(x_sample, u_sample)
        _,_,V = np.linalg.svd(A)
        e = V.T[:,-1]

        u_test = u_full[arg_thresh[0][test_indices], :]
        x_test = x_full[arg_thresh[0][test_indices], :]
        
        dist = np.cross(x_test, u_test)@e
        idx = np.hstack((sample_indices.flatten(), test_indices[abs(dist)<eps].flatten()))
        inliers  = arg_thresh[0][idx]
        

        #NOTE: inliers need to be indices in original input (unthresholded), 
        #sample indices before test indices for the autograder
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_ep = e
            best_inliers = inliers

    return best_ep, best_inliers