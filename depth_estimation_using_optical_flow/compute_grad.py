import numpy as np
import scipy

"""
Define 1D filters as global variables.
"""
g = [0.015625, 0.093750, 0.234375, 0.312500, 0.234375, 0.093750, 0.015625]
h = [0.03125, 0.12500, 0.15625, 0, -0.15625, -0.1250, -0.03125]
def compute_Ix(imgs):
    """
    params:
        @imgs: np.array(h, w, N)
    return value:
        Ix: np.array(h, w, N)
    """
    
    Ix = scipy.ndimage.convolve1d(imgs, weights=h, axis=1)
    Ix = scipy.ndimage.convolve1d(Ix, weights=g, axis=0)
    Ix = scipy.ndimage.convolve1d(Ix, weights=g, axis=2)
    return Ix

def compute_Iy(imgs):
    """
    params:
        @imgs: np.array(h, w, N)
    return value:
        Iy: np.array(h, w, N)
    """
    Iy = scipy.ndimage.convolve1d(imgs, weights=h, axis=0)
    Iy = scipy.ndimage.convolve1d(Iy, weights=g, axis=1)
    Iy = scipy.ndimage.convolve1d(Iy, weights=g, axis=2)
    return Iy

def compute_It(imgs):
    """
    params:
        @imgs: np.array(h, w, N)
    return value:
        It: np.array(h, w, N)
    """
    It = scipy.ndimage.convolve1d(imgs, weights=h, axis=2)
    It = scipy.ndimage.convolve1d(It, weights=g, axis=0)
    It = scipy.ndimage.convolve1d(It, weights=g, axis=1)
    return It
