import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points
    """
    points = np.array((
        (0, 0, 1),
        (width, 0, 1),
        (0, height, 1),
        (width, height, 1),
    ), dtype=np.float32).reshape(2, 2, 3)
    
    points[0,0,:] = np.linalg.pinv(K)@points[0,0,:]
    points[0,1,:] = np.linalg.pinv(K)@points[0,1,:]
    points[1,0,:] = np.linalg.pinv(K)@points[1,0,:]
    points[1,1,:] = np.linalg.pinv(K)@points[1,1,:]

    points = depth*points
    
    points[0,0,:] = (np.linalg.pinv(Rt[:,:3])@points[0,0,:])- np.linalg.pinv(Rt[:,:3])@Rt[:,3]
    points[0,1,:] = (np.linalg.pinv(Rt[:,:3])@points[0,1,:]) - np.linalg.pinv(Rt[:,:3])@Rt[:,3]
    points[1,0,:] = (np.linalg.pinv(Rt[:,:3])@points[1,0,:])[:3]- np.linalg.pinv(Rt[:,:3])@Rt[:,3]
    points[1,1,:] = (np.linalg.pinv(Rt[:,:3])@points[1,1,:])[:3]- np.linalg.pinv(Rt[:,:3])@Rt[:,3]

    return points

def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """

    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            points[i,j,:] = K@(Rt[:,:3]@points[i,j,:] + Rt[:,3])
            points[i,j,:] = points[i,j,:]/points[i,j,2]

    return points[:,:,:2]

def warp_neighbor_to_ref(backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor):
    """ 
    Warp the neighbor view into the reference view 
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]

    points = backproject_fn(K_ref, width, height, depth, Rt_ref)
    points_n = project_fn(K_neighbor, Rt_neighbor, points).reshape((4,2))
    #print(points_n.shape)
    points_ref = project_fn(K_ref, Rt_ref, points).reshape((4,2))
    H, _ = cv2.findHomography(points_n, points_ref)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, H, (width, height))

  
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """ 
    Compute the cost map between src and dst patchified images via the ZNCC metric
    
    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value, 
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    num_R = np.sum((src[:,:,:,0] - np.mean(src[:,:,:,0], axis=2)[:,:,None]) * (dst[:,:,:,0] - np.mean(dst[:,:,:,0], axis=2)[:,:,None]), axis=2)
    den_R = (np.std(src[:,:,:,0],axis=2)+EPS) * (np.std(dst[:,:,:,0], axis=2) + EPS)
    zncc_R = num_R/den_R
    #print(zncc_R.shape)
    num_G = np.sum((src[:,:,:,1] - np.mean(src[:,:,:,1], axis=2)[:,:,None]) * (dst[:,:,:,1] - np.mean(dst[:,:,:,1], axis=2)[:,:,None]), axis=2)
    den_G = (np.std(src[:,:,:,1],axis=2)+EPS) * (np.std(dst[:,:,:,1], axis=2) + EPS)
    zncc_G = num_G/den_G
    num_B = np.sum((src[:,:,:,2] - np.mean(src[:,:,:,2], axis=2)[:,:,None]) * (dst[:,:,:,2] - np.mean(dst[:,:,:,2], axis=2)[:,:,None]), axis=2)
    den_B = (np.std(src[:,:,:,2],axis=2)+EPS) * (np.std(dst[:,:,:,2], axis=2) + EPS)
    zncc_B = num_B/den_B
    zncc = zncc_R + zncc_G + zncc_B

    return zncc  # height x width


def backproject(dep_map, K):
    """ 
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))


    xyz_cam = np.zeros((dep_map.shape[0], dep_map.shape[1], 3))
    xyz_pix = np.ones_like(xyz_cam)
    xyz_pix[:,:,0] = _u
    xyz_pix[:,:,1] = _v
    for i in range(xyz_pix.shape[0]):
        for j in range(xyz_pix.shape[1]):
            xyz_cam[i,j,:] = np.linalg.pinv(K)@xyz_pix[i,j,:]
    xyz_cam = xyz_cam*dep_map[:,:,None]

    return xyz_cam

