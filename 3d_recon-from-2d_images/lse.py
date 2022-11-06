import numpy as np

def least_squares_estimation(X1, X2):

  N = np.shape(X1)[0]
  A = np.zeros((N, 9))
  for i in range(N):
    A[i,:] = np.hstack((X2[i,0]*X1[i,:], X2[i,1]*X1[i,:], X2[i,2]*X1[i,:]))
  
  u, s, v = np.linalg.svd(A)
  e = v.T[:,-1]
  U, S, V = np.linalg.svd(e.reshape(3,3))
  diag = np.identity(3)
  diag[2][2] = 0
  E = U @ diag @ V

  return E
