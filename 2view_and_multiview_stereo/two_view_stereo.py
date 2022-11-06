import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding


    H_i = K_i_corr @ R_irect @ np.linalg.pinv(K_i) 
    H_j = K_j_corr @ R_jrect @ np.linalg.pinv(K_j)

    rgb_i_rect = cv2.warpPerspective(rgb_i, H_i, dsize = (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_j, dsize = (w_max, h_max))
    #print(rgb_i_rect.shape, rgb_j_rect.shape)

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    R_wi, R_wj : [3,3]
    T_wi, T_wj : [3,1]
        p_i = R_wi @ p_w + T_wi
        p_j = R_wj @ p_w + T_wj
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ji @ p_j + T_ji, B is the baseline
    """

    center_pi = np.zeros((3,1))
    center_pj = np.zeros((3,1))
    R_ji = R_wi@np.linalg.pinv(R_wj)
    T_ji = -R_wi@np.linalg.pinv(R_wj)@T_wj + T_wi
    center_wi = np.linalg.pinv(R_wi)@(center_pi - T_wi)
    center_wj = np.linalg.pinv(R_wj)@(center_pj - T_wj)
    B = np.linalg.norm(center_wi - center_wj)
    return R_ji, T_ji, B


def compute_rectification_R(T_ji):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ji : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ji.squeeze(-1) / (T_ji.squeeze(-1)[1] + EPS)

    opt_axis = np.array([0,0,1])
    r2 = e_i
    r1 = np.cross(r2, opt_axis)
    r3 = np.cross(r1, r2)
    r1 /= np.linalg.norm(r1)
    r2 /= np.linalg.norm(r2)
    r3 /= np.linalg.norm(r3)
    #print(r2.shape, r1.shape, r3.shape)
    R_irect = np.vstack((r1.T, r2.T, r3.T))
    
    #print(R_irect.shape)
    return R_irect


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    src = src[:,None,:,:]
    dst = dst[None,:,:,:]
    #print(src.shape)
    #print(dst.shape)
    sum_pix = np.sum((src-dst)**2, axis=2)
    #print(sum_pix.shape)
    ssd = np.sum(sum_pix, axis=2)
    #print(ssd.shape)
    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    src = src[:,None,:,:]
    dst = dst[None,:,:,:]
    #print(src.shape)
    #print(dst.shape)
    sum_pix = np.sum(np.abs(src-dst), axis=2)
    #print(sum_pix.shape)
    sad = np.sum(sum_pix, axis=2)
    #print(ssd.shape)
    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    R_src = src[:,:,0]
    G_src = src[:,:,1]
    B_src = src[:,:,2]

    R_dst = dst[:,:,0]
    G_dst = dst[:,:,1]
    B_dst = dst[:,:,2]

    #print(R_src.shape)
    src_mean_R = R_src - np.mean(R_src, axis=1)[:,None]
    dst_mean_R = R_dst - np.mean(R_dst, axis=1)[:,None]

    src_mean_G = G_src - np.mean(G_src, axis=1)[:,None]
    dst_mean_G = G_dst - np.mean(G_dst, axis=1)[:,None]

    src_mean_B = B_src - np.mean(B_src, axis=1)[:,None]
    dst_mean_B = B_dst - np.mean(B_dst, axis=1)[:,None]

    #print(src_mean_R.shape)
    temp_R = np.zeros((src.shape[0], dst.shape[0]))
    temp_G = np.zeros((src.shape[0], dst.shape[0]))
    temp_B = np.zeros((src.shape[0], dst.shape[0]))
    for i in range(src.shape[0]):
        for j in range(dst.shape[0]):
            # print((src_mean_R[i]*dst_mean_R[j]).shape)
            temp_R[i,j] = np.sum(src_mean_R[i]*dst_mean_R[j])
            temp_G[i,j] = np.sum(src_mean_G[i]*dst_mean_G[j])
            temp_B[i,j] = np.sum(src_mean_B[i]*dst_mean_B[j])
    
    den_R = (np.std(R_src,axis=1)+EPS) * (np.std(R_dst, axis=1) + EPS)
    den_G = (np.std(G_src,axis=1)+EPS) * (np.std(G_dst, axis=1) + EPS)
    den_B = (np.std(B_src,axis=1)+EPS) * (np.std(B_dst, axis=1) + EPS)

    # print(temp_R.shape)
    # print(den_R.shape)
    zncc_R = temp_R/den_R
    zncc_G = temp_G/den_G
    zncc_B = temp_B/den_B

    zncc = zncc_R + zncc_G + zncc_B
    #print(zncc.shape)
    
    #######################################
    # src_mean_R = src[:,:,0] - np.mean(src[:,:,0], axis=1)[:,None]
    # dst_mean_R = dst[:,:,0] - np.mean(dst[:,:,0], axis=1)[:,None]
    # temp_num_R = np.zeros((src.shape[0], dst.shape[0], src.shape[1]))
    # for i in range(src_mean_R.shape[0]):
    #     temp_num_R[i,:,:] = src_mean_R[i,:]*dst_mean_R[:,:]
    # num_R = np.sum(temp_num_R, axis=2)
    # den_R = (np.std(src[:,:,0],axis=1)+EPS) * (np.std(dst[:,:,0], axis=1) + EPS)
    # zncc_R = num_R/den_R
    # #print(zncc_R.shape)

    # src_mean_G = src[:,:,1] - np.mean(src[:,:,1], axis=1)[:,None]
    # dst_mean_G = dst[:,:,1] - np.mean(dst[:,:,1], axis=1)[:,None]
    # temp_num_G = np.zeros((src.shape[0], dst.shape[0], src.shape[1]))
    # for i in range(src_mean_G.shape[0]):
    #     temp_num_G[i,:,:] = src_mean_G[i,:]*dst_mean_G[:,:]
    # num_G = np.sum(temp_num_G, axis=2)
    # den_G = (np.std(src[:,:,1],axis=1)+EPS) * (np.std(dst[:,:,1], axis=1) + EPS)
    # zncc_G = num_G/den_G

    # src_mean_B = src[:,:,2] - np.mean(src[:,:,2], axis=1)[:,None]
    # dst_mean_B = dst[:,:,2] - np.mean(dst[:,:,2], axis=1)[:,None]
    # temp_num_B = np.zeros((src.shape[0], dst.shape[0], src.shape[1]))
    # for i in range(src_mean_B.shape[0]):
    #     temp_num_B[i,:,:] = src_mean_B[i,:]*dst_mean_B[:,:]
    # num_B = np.sum(temp_num_B, axis=2)
    # den_B = (np.std(src[:,:,2],axis=1)+EPS) * (np.std(dst[:,:,2], axis=1) + EPS)
    # zncc_B = num_B/den_B
    # # num_G = (src[:,:,1] - np.mean(src[:,:,1], axis=1)[:,None]) @ (dst[:,:,1] - np.mean(dst[:,:,1], axis=1)[:,None]).T
    # # den_G = (np.std(src[:,:,1],axis=1)+EPS) * (np.std(dst[:,:,1], axis=1) + EPS)
    # # zncc_G = num_G/den_G
    # # num_B = (src[:,:,2] - np.mean(src[:,:,2], axis=1)[:,None]) @ (dst[:,:,2] - np.mean(dst[:,:,2], axis=1)[:,None]).T
    # # den_B = (np.std(src[:,:,2],axis=1)+EPS) * (np.std(dst[:,:,2], axis=1) + EPS)
    # # zncc_B = num_B/den_B
    # zncc = zncc_R + zncc_G + zncc_B

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    patch_buffer = np.zeros((image.shape[0], image.shape[1], k_size**2, 3))
    img_pad = np.pad(image, ((k_size//2, k_size//2), (k_size//2, k_size//2), (0,0) ), 'constant')
    #print(patch_buffer.shape)
    #print(img_pad.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            #print(k_size//2 + i)
            #print(k_size//2 + j)
            temp_R = img_pad[k_size//2 + i - k_size//2:k_size//2 + i + k_size//2 +1, k_size//2 + j - k_size//2:k_size//2 + j + k_size//2 +1, 0]
            temp_G = img_pad[k_size//2 + i - k_size//2:k_size//2 + i + k_size//2 +1, k_size//2 + j - k_size//2:k_size//2 + j + k_size//2 +1, 1]
            temp_B = img_pad[k_size//2 + i - k_size//2:k_size//2 + i + k_size//2 +1, k_size//2 + j - k_size//2:k_size//2 + j + k_size//2 +1, 2]
            #print(temp_R.shape)
            #print(temp_G.shape)
            #print(temp_B.shape)
            patch_buffer[i,j,:,0] = temp_R.flatten()
            patch_buffer[i,j,:,1] = temp_G.flatten()
            patch_buffer[i,j,:,2] = temp_B.flatten()
            
    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """
    
    h, w = rgb_i.shape[:2]
    #print(h,w)
    disp_map = np.zeros((h,w), dtype=np.float64)
    lr_consistency_mask = np.zeros((h,w), dtype=np.float64)

    patches_i = image2patch(rgb_i.astype(float) / 255.0, k_size)  # [h,w,k*k,3]
    patches_j = image2patch(rgb_j.astype(float) / 255.0, k_size)  # [h,w,k*k,3]
    vi_idx, vj_idx = np.arange(h), np.arange(h)
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
    #print(disp_candidates.shape)
    valid_disp_mask = disp_candidates > 0.0
    #print(valid_disp_mask.shape)
    for u in range(rgb_i.shape[1]):
        
        buf_i, buf_j = patches_i[:, u], patches_j[:, u]
        value = kernel_func(buf_i, buf_j)  # each row is one pix from left, col is the disparity
        #print(value.shape)
        _upper = value.max() + 1.0
        value[~valid_disp_mask] = _upper
        #best_matched_right_pixel = value[v].argmin()
        best_matched_right_pixel = np.argmin(value, axis=1)
        #best_matched_left_pixel = value[:,best_matched_right_pixel].argmin()
        best_matched_left_pixel = np.argmin(value[:,best_matched_right_pixel.flatten()], axis=0)
        consistent_flag = best_matched_left_pixel == vi_idx
        #print(u)
        # print(best_matched_right_pixel.shape)
        # print(disp_candidates[:][best_matched_right_pixel].shape)
        # print(disp_map.shape)
        # print(disp_map[:, u].shape)
        disp_map[:,u] = np.take_along_axis(disp_candidates, np.expand_dims(best_matched_right_pixel, axis=1), axis=1).flatten()
        
        
        lr_consistency_mask[consistent_flag,u] = 1.0

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    center_x = K[0,2]
    center_y = K[1,2]
    xyz_cam = np.zeros((disp_map.shape[0], disp_map.shape[1], 3), dtype=np.float64)
    x, y = np.meshgrid(np.arange(disp_map.shape[1]), np.arange(disp_map.shape[0]))

    dep_map = B*K[1,1]*1.0/disp_map
    #print(dep_map.shape)
    xyz_cam[:,:,2] = dep_map
    xyz_cam[:,:,0] = ((x - center_x)*dep_map)*1.0/K[0,0]
    xyz_cam[:,:,1] = ((y - center_y)*dep_map)*1.0/K[1,1]

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_wc,
    T_wc,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    pcl_world = np.zeros_like(pcl_cam)
    # for i in range(pcl_cam.shape[0]):
    #     pcl_world[i,:] = (np.linalg.pinv(R_wc)@(pcl_cam[i,:].reshape((3,1)) - T_wc)).flatten()
    pcl_world = np.linalg.pinv(R_wc)@(pcl_cam.T - T_wc)
    #pcl_world = R_wc.T@pcl_cam.T - R_wc.T@T_wc.reshape(-1,1)
    #print((pcl_world.T).shape)

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world.T, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_wi, T_wi = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_wj, T_wj = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ji, T_ji, B = compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj)
    assert T_ji[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ji)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ji,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_wc=R_irect @ R_wi,
        T_wc=R_irect @ T_wi,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
