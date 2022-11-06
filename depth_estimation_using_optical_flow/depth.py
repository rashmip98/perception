import numpy as np

def depth(flow, confidence, ep, K, thres=10):
    """
    params:
        @flow: np.array(h, w, 2)
        @confidence: np.array(h, w, 2)
        @K: np.array(3, 3)
    return value:
        depth_map: np.array(h, w)
    """
    depth_map = np.zeros_like(confidence)


    u = flow[:,:,0]
    v = flow[:,:,1]
    x, y = np.meshgrid(np.linspace(0, u.shape[0]-1, u.shape[0]), np.linspace(0 , u.shape[1] -1, u.shape[1]))
    xp = np.vstack((x.flatten().reshape(1,-1), y.flatten().reshape(1,-1), np.ones(x.flatten().shape[0]).reshape(1,-1) ))
    
    p_trans = np.vstack((u.flatten().reshape(1,-1), v.flatten().reshape(1,-1), np.zeros(u.flatten().shape[0]) ))
    #p_trans = np.linalg.inv(K)@up
    p_trans[0,:] = p_trans[0,:]/K[0,0]
    p_trans[1,:] = p_trans[1,:]/K[1,1]
    # ep_calibrated = np.zeros(3)
    # ep_calibrated[0] = ep[0]/K[0,0]
    # ep_calibrated[1] = ep[1]/K[1,1]
    ep_calibrated = np.linalg.inv(K)@ep
    xp_calibrated = np.linalg.inv(K)@xp
    #d = np.linalg.norm(p_trans[:2,:], axis=0)/np.linalg.norm(xp_calibrated[:2,:] - ep_calibrated[:2].reshape(-1,1), axis=0)
    d = np.linalg.norm(xp_calibrated[:-1,:] - ep_calibrated[:-1].reshape(-1,1), axis=0)/np.linalg.norm(p_trans, axis=0)
    depth_map = d.reshape(depth_map.shape[0], depth_map.shape[1])
    depth_map = np.where(confidence>thres,depth_map,0)
    truncated_depth_map = np.maximum(depth_map, 0)
    valid_depths = truncated_depth_map[truncated_depth_map > 0]
    # You can change the depth bound for better visualization if your depth is in different scale
    depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)
    print(f'depth bound: {depth_bound}')

    truncated_depth_map[truncated_depth_map > depth_bound] = 0
    truncated_depth_map = truncated_depth_map / truncated_depth_map.max()


    return truncated_depth_map
