# Code Functionality:

- scc_module : main implementation of the supporting functions with implementation and codes for all the NCC and SCC functions

- window_nodes : creates required windows (matrix of values of pixels around the center value of a node as passed in the function). Standalone run will result in a visualization of the selected nodes windows

- student_gaussian_apodization : Gaussian apodization function and scc algorithm code here. 

- GroundTruthPlotter : plots the ground truth (the boundary of the synthetic data)

- boundary_util : plots the boundary and the nodes associated with the boundary as per the synthetic data

- executioner : program to run all the designated functions and keep scc_module code cleaner

Results folder added for providing important results directly.
