import numpy as np
import pdb

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
        @x: int
        @y: int
    return value:
        flow: np.array(2,)
        conf: np.array(1,)
    """

    xmin = np.clip(x- (size//2), 0, Ix.shape[1]-1)
    xmax = np.clip(x+ (size//2), 0, Ix.shape[1]-1)
    ymin = np.clip(y- (size//2), 0, Ix.shape[1]-1)
    ymax = np.clip(y+ (size//2), 0, Ix.shape[1]-1)
    A = np.hstack((Ix[ymin:ymax+1,xmin:xmax+1].reshape(-1,1), Iy[ymin:ymax+1,xmin:xmax+1].reshape(-1,1)))
    b = -1*It[ymin:ymax+1,xmin:xmax+1].reshape(-1,1)
  
    #print(A.shape, b.shape)
    flow,_,_, conf = np.linalg.lstsq(A,b, rcond=None)  
    flow = flow.flatten()
    conf_arr = conf.flatten()
    conf = np.min(conf_arr)
    #print(conf.shape)
    return flow, conf


def flow_lk(Ix, Iy, It, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
    return value:
        flow: np.array(h, w, 2)
        conf: np.array(h, w)
    """
    image_flow = np.zeros([Ix.shape[0], Ix.shape[1], 2])
    confidence = np.zeros([Ix.shape[0], Ix.shape[1]])
    for x in range(Ix.shape[1]):
        for y in range(Ix.shape[0]):
            flow, conf = flow_lk_patch(Ix, Iy, It, x, y)
            image_flow[y, x, :] = flow
            confidence[y, x] = conf
    return image_flow, confidence

    

