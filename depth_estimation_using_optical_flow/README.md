## Description
* Compute the spatiotemporal gradients Ix, Iy, It
* Assuming that the optical flow is constant in a local neighborhood of 5x5 pixels, the optical flow field can be computed using the 25x2 linear system consisting of 25 equations of the form: Ixu + Iyv + It = 0 where (Ix, Iy, It) are the spatiotemporal derivatives at every pixel and (u,v) are the flow components
* Compute the pixel position of the epipole using the optical flow and every pixel's position
* Given pixel flow, confidence, epipole, and intrinsic parameters compute a depth at every pixel for which flow exists
