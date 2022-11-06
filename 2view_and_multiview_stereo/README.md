## Description
#### Two-view Stereo
* The main file for this is the two_view.ipynb file
* We first rectify the views, then compute disparity map, then compute depth map and point cloud, then postprocess to remove the background, crop the depth map and remove the point cloud out-liers.

#### Plane-sweep Stereo
* The main file for this is the plane_sweep.ipynb
* We first warp the neighboring views, construct a cost map, then construct the cost volume. We can obtain a pointcloud from our computed depth map, by backprojecting the reference image pixels to their corresponding depths, which produces a series of 3D coordinates with respect to the camera frame.

A reconstructed output is given [here](https://github.com/rashmip98/perception/blob/main/2view_and_multiview_stereo/recon_2view.png)
