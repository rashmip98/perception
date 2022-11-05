import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    f = (K[0][0] + K[1][1])/2.0
    pix_center = np.array([[K[0][2], K[1][2]]])

    u_v = (Pc + pix_center)

    #alpha = np.dot(Pw[2,:],Pw[3,:])/(np.linalg.norm(Pw[2,:])*np.linalg.norm(Pw[3,:]))
    #beta = np.dot(Pw[0,:],Pw[3,:])/(np.linalg.norm(Pw[0,:])*np.linalg.norm(Pw[3,:]))
    #gamma = np.dot(Pw[0,:],Pw[2,:])/(np.linalg.norm(Pw[0,:])*np.linalg.norm(Pw[2,:]))

    c = np.linalg.norm(Pw[1,:] - Pw[2,:])
    a = np.linalg.norm(Pw[2,:] - Pw[3,:])
    b = np.linalg.norm(Pw[1,:] - Pw[3,:])

    j1 = (1/(np.sqrt(u_v[1,0]**2+u_v[1,1]**2+f**2))) * np.array([[u_v[1,0], u_v[1,1], f]])
    j2 = (1/(np.sqrt(u_v[2,0]**2+u_v[2,1]**2+f**2))) * np.array([[u_v[2,0], u_v[2,1], f]])
    j3 = (1/(np.sqrt(u_v[3,0]**2+u_v[3,1]**2+f**2))) * np.array([[u_v[3,0], u_v[3,1], f]])

    alpha = np.dot(j2,j3.T)
    beta = np.dot(j1,j3.T)
    gamma = np.dot(j1,j2.T)
    #print(j2,j3,alpha)
    A4 = ((a**2 - c**2)/b**2 - 1)**2 - (4*c**2*alpha**2/b**2)
    A3 = 4*(((a**2 - c**2)/b**2)*(1 - ((a**2 - c**2)/b**2))*beta - ((1 - ((a**2 - c**2)/b**2))*alpha*gamma) + ((2*c**2*alpha**2*beta)/b**2))
    A2 = 2*( ((a**2 - c**2)/b**2)**2 -1 + 2*((a**2 - c**2)/b**2)**2*beta**2 + 2*((b**2 - c**2)/b**2)*alpha**2 - 4*((a**2 + c**2)/b**2)*alpha*beta*gamma + 2*((b**2 - a**2)/b**2)*gamma**2 )
    A1 = 4*( -1*((a**2 - c**2)/b**2)*(1 + ((a**2 - c**2)/b**2))*beta + (2*a**2*gamma**2*beta)/b**2 - (1 - ((a**2 + c**2)/b**2))*alpha*gamma )
    A0 = ( 1 + ((a**2 - c**2)/b**2))**2 - (4*a**2*gamma**2)/b**2

    coeff = [A4.item(), A3.item(), A2.item(), A1.item(), A0.item()]
    roots = np.roots(coeff)

    for v in roots:
        u = ((-1 + ((a**2 - c**2)/b**2))*v**2 - 2*((a**2 - c**2)/b**2)*beta*v + 1 + ((a**2 - c**2)/b**2))/ ( 2*(gamma - v*alpha) )
        #print(u,v)
        if 0<np.abs(u.imag)<1 and 0<np.abs(v.imag) < 1 and u.real>0 and v.real>0 and not(np.isinf(u)):
            s1 = np.sqrt(a**2/(u.real**2 + v.real**2 - 2*u.real*v.real*alpha))
            s2 = u.real*s1
            s3 = v.real*s1
            #print(s1,s2,s3)
            break

    X = np.squeeze(np.stack([s1*j1, s2*j2, s3*j3], axis = 0))
    #print(X.shape)    
    R, t = Procrustes(X, Pw)
    
    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    B = Y[1:4,:]
    A = X[0:3,:]
    U, S, V = np.linalg.svd(B@A.T)
    diag = np.array(
        [[1, 0, 0],
        [0, 1, 0],
        [0, 0, np.linalg.det(V.T@U.T)]]
    )
    R = V.T@diag@U.T
    t = np.sum(X, axis=0)/X.shape[0] - R@( np.sum(B, axis=0)/B.shape[0] )

    return R, t
