from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    # Homography Approach
    # Pose from Projective Transformation
    H = est_homography(Pw[:,0:2],Pc)
    K_inv_H = np.linalg.inv(K)@H
    K_inv_H_updated = np.hstack((K_inv_H[:,0:2], np.cross(K_inv_H[:,0],K_inv_H[:,1]).reshape(K_inv_H.shape[0],1)))
    u, s, v = np.linalg.svd(K_inv_H_updated)
    v_trans = v.T
    diag = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, np.linalg.det(u@v)]
    ])
    R_old = u@diag@v
    t_old = K_inv_H[:,-1]/np.linalg.norm(K_inv_H_updated[:,0])
    R = np.linalg.inv(R_old)
    t = -1*R@t_old

    return R, t