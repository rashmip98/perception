import numpy as np

def pose_candidates_from_E(E):
  transform_candidates = []
  ##Note: each candidate in the above list should be a dictionary with keys "T", "R"

  U, S, V = np.linalg.svd(E)
  Rz90 = np.array([[0,-1,0],[1,0,0],[0,0,1]])
  Rzneg90 = np.array([[0,1,0],[-1,0,0],[0,0,1]])
  R1 = U@Rz90.T@V
  R2 = U@Rzneg90.T@V
  T1 = U[:,-1]
  T2 = -1*U[:,-1]
  transform_candidates = [{'T':T1,'R':R1}, {'T':T1,'R':R2}, {'T':T2,'R':R1}, {'T':T2,'R':R2} ]

  return transform_candidates