import numpy as np

def compute_planar_params(flow_x, flow_y, K,
                                up=[256, 0], down=[512, 256]):
    """
    params:
        @flow_x: np.array(h, w)
        @flow_y: np.array(h, w)
        @K: np.array(3, 3)
    return value:
        sol: np.array(8,)
    """
    flow_x_rect = flow_x[up[0]:down[0], up[1]:down[1]]/K[0,0]
    flow_y_rect = flow_y[up[0]:down[0], up[1]:down[1]]/K[1,1]
    x_rect, y_rect = np.meshgrid(np.linspace(up[0], down[0] -1, flow_x_rect.shape[0]), np.linspace(up[1] , down[1]-1, flow_x_rect.shape[1]))
    x, y = np.meshgrid(np.linspace(0, flow_x.shape[0] -1, flow_x.shape[0]), np.linspace(0 , flow_x.shape[1]-1, flow_x.shape[1]))
    # x, y = np.meshgrid(np.linspace(-flow_x.shape[0]/2, flow_x.shape[0]/2 -1, flow_x.shape[0]), np.linspace(-flow_x.shape[1]/2 , flow_x.shape[1]/2 -1, flow_x.shape[1]))
    # x_rect = np.array([i for i in range(up[0],down[0])])
    # y_rect= np.array([i for i in range(up[1], down[1])])
    x_rect = x[up[0]:down[0], up[1]:down[1]]
    y_rect = y[up[0]:down[0], up[1]:down[1]]
    
    xp = np.vstack((x_rect.reshape(1,-1), y_rect.reshape(1,-1), np.ones(x_rect.flatten().shape[0]).reshape(1,-1)))
    xp_calibrated = np.linalg.inv(K)@xp
    b = np.vstack((flow_x_rect.flatten().reshape(-1,1), flow_y_rect.flatten().reshape(-1,1)))
    #A_upper = np.hstack((np.ones(flow_x_rect.flatten().shape[0]).reshape(-1,1), xp_calibrated[0].flatten().reshape(-1,1), xp_calibrated[1].flatten().reshape(-1,1), np.zeros((flow_x_rect.flatten().shape[0], 3)), (xp_calibrated[0]**2).flatten().reshape(-1,1), (xp_calibrated[0]*xp_calibrated[1]).flatten().reshape(-1,1) ))
    #A_lower = np.hstack((np.zeros((flow_x_rect.flatten().shape[0], 3)), np.ones(flow_x_rect.flatten().shape[0]).reshape(-1,1), xp_calibrated[0].flatten().reshape(-1,1), xp_calibrated[1].flatten().reshape(-1,1), (xp_calibrated[0]*xp_calibrated[1]).flatten().reshape(-1,1), (xp_calibrated[1]**2).flatten().reshape(-1,1) ))

    A_upper_reord = np.hstack(((xp_calibrated[0]**2).flatten().reshape(-1,1), (xp_calibrated[0]*xp_calibrated[1]).flatten().reshape(-1,1), xp_calibrated[0].flatten().reshape(-1,1), xp_calibrated[1].flatten().reshape(-1,1), np.ones(flow_x_rect.flatten().shape[0]).reshape(-1,1), np.zeros((flow_x_rect.flatten().shape[0], 3)) ))
    A_lower_reord = np.hstack(((xp_calibrated[0]*xp_calibrated[1]).flatten().reshape(-1,1), (xp_calibrated[1]**2).flatten().reshape(-1,1), np.zeros((flow_x_rect.flatten().shape[0], 3)), xp_calibrated[1].flatten().reshape(-1,1), xp_calibrated[0].flatten().reshape(-1,1), np.ones(flow_x_rect.flatten().shape[0]).reshape(-1,1) ))
    #A = np.vstack((A_upper, A_lower))
    #sol = np.linalg.pinv(A)@b
    A_reord = np.vstack((A_upper_reord, A_lower_reord))
    sol = np.linalg.pinv(A_reord)@b
    #print(sol_reord)
    #i = [4,2,3,7,6,5,0,1] #3rd - not 1,3
    #i = [6,7,5, 1,2,0,3,4]
    #i = [6,7,4, 2,1,0,5,3]
    #q_x = np.linalg.pinv(A_upper_reord)@flow_x_rect.flatten().reshape(-1,1)
    #q_y = np.linalg.pinv(A_lower_reord)@flow_y_rect.flatten().reshape(-1,1)
    #i = [6,7,1, 2,0,5,4,3]
    #print(q_x)
    #print(q_y)
    #sol = sol[i,:]
    # q,_,_,_ = np.linalg.lstsq(A,b, rcond=None)
    # #print(q.shape)
    # print(q)
    sol = sol.flatten()
    # A_x = np.hstack(((np.square(xp_calibrated[0])).flatten().reshape(-1,1), (xp_calibrated[0]*xp_calibrated[1]).flatten().reshape(-1,1), xp_calibrated[0].flatten().reshape(-1,1), xp_calibrated[1].flatten().reshape(-1,1), np.ones(flow_x_rect.flatten().shape[0]).reshape(-1,1) ))
    # # q_x = np.linalg.pinv(A_x)@flow_x_rect.flatten().reshape(-1,1)
    # A_y = np.hstack(((np.square(xp_calibrated[1])).flatten().reshape(-1,1), (xp_calibrated[0]*xp_calibrated[1]).flatten().reshape(-1,1), xp_calibrated[1].flatten().reshape(-1,1), xp_calibrated[0].flatten().reshape(-1,1), np.ones(flow_y_rect.flatten().shape[0]).reshape(-1,1) ))
    # # q_y = np.linalg.pinv(A_y)@flow_y_rect.flatten().reshape(-1,1)
    # q_x,_,_,_ = np.linalg.lstsq(A_x, flow_x_rect.flatten().reshape(-1,1), rcond=None)
    # q_y,_,_,_ = np.linalg.lstsq(A_y, flow_y_rect.flatten().reshape(-1,1), rcond=None)
    # sol = np.zeros(8)
    # print(A_x.shape)
    # print(A_y.shape)
    # print(q_x[0]==q_y[1])
    # # sol[:5] = q_x.flatten()
    # # sol[5:] = q_y[2:].flatten()
    # print(q_x) 
    # print(q_y)
    return sol
    




