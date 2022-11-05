import numpy as np

def est_homography(X, Y):
    """
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out
    what X and Y should be.
    Input:
        X: 4x2 matrix of (x,y) coordinates
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    H = []
    A = np.array([
          [X[0,0], X[0,1], 1, 0, 0, 0, -X[0,0]*Y[0,0], -X[0,1]*Y[0,0], -Y[0,0]],
          [0, 0, 0, X[0,0], X[0,1], 1, -X[0,0]*Y[0,1], -X[0,1]*Y[0,1], -Y[0,1]],
          [X[1,0], X[1,1], 1, 0, 0, 0, -X[1,0]*Y[1,0], -X[1,1]*Y[1,0], -Y[1,0]],
          [0, 0, 0, X[1,0], X[1,1], 1, -X[1,0]*Y[1,1], -X[1,1]*Y[1,1], -Y[1,1]],
          [X[2,0], X[2,1], 1, 0, 0, 0, -X[2,0]*Y[2,0], -X[2,1]*Y[2,0], -Y[2,0]],
          [0, 0, 0, X[2,0], X[2,1], 1, -X[2,0]*Y[2,1], -X[2,1]*Y[2,1], -Y[2,1]],
          [X[3,0], X[3,1], 1, 0, 0, 0, -X[3,0]*Y[3,0], -X[3,1]*Y[3,0], -Y[3,0]],
          [0, 0, 0, X[3,0], X[3,1], 1, -X[3,0]*Y[3,1], -X[3,1]*Y[3,1], -Y[3,1]]])

    u, s, v = np.linalg.svd(A)
    v_trans = v.T
    h_norm = v_trans[:,-1]/v_trans[-1,-1]
    H = h_norm.reshape(3,3)

    return H
