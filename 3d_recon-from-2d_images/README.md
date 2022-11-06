## Description
We recover the 3D transformation (R, T) between the two views, such that P1, P2 ∈ R^3 describe the same scene point in frame 1 and 2, and P2 = RP1 + T

The pipeline goes like this:
* Estimate the Essential matrix E using the method of SVD decomposition in the 8-pt algorithm 
* Compute the four possible solutions for (R, T) i.e. the twisted pair for E and the twisted pair for -E
* Select the right (T, R) pair out of the 4 possibilities

## Usage
The main.ipynb file calls the functions from the other files
