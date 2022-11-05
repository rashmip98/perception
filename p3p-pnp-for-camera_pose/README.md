## Description
Given a video with an AprilTags (https://april.eecs.umich.edu/software/apriltag) in each frame, we first recover the camera pose estimation by either solving the Perspective-N-Point (PnP) problem with coplanar assumption
OR by solving the Persepective-three-point (P3P) and the Procrustes problem.
After retrieving the 3D relationship between the camera and world, we can place any arbitrary objects in the scene.
The final output will be something as follows:

![image](https://user-images.githubusercontent.com/31537022/200145704-de8aeafd-2ffa-41bb-a18f-7b831041bdd0.png)

Here, two virtual object models of a Drill and a Bottle are placed in the image as if they exist.
