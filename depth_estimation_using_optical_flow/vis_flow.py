import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    params:
        @img: np.array(h, w)
        @flow_image: np.array(h, w, 2)
        @confidence: np.array(h, w)
    return value:
        None
    """

    """
    STUDENT CODE BEGINS
    """
    #print((confidence>threshmin))
    x = []
    y = []
    flow_x = []
    flow_y = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if confidence[i,j]>threshmin:
                x.append(j)
                y.append(511-i)
                flow_x.append(flow_image[i,j,1])
                flow_y.append(flow_image[i,j,0])
    #image = np.where(confidence>threshmin, image,0)
    #flow_image = np.where(confidence>threshmin, flow_image)
    #print(image.shape)
    #x, y = np.meshgrid(np.linspace(-image.shape[0], image.shape[0], 1), np.linspace(-image.shape[1], image.shape[1], 1))
    #flow_x, flow_y = flow_image[x]
    
    plt.quiver(x, y, (np.asarray(flow_x)*10).astype(int), (np.asarray(flow_y)*10).astype(int), angles='xy', scale_units='xy', scale=1., color='red', width=0.001)

    
    """
    STUDENT CODE ENDS
    """
    return





    

