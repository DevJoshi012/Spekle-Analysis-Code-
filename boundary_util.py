#boundary utility file converts and develops the boundary for each frame:

#essential imports
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

#important files to load : im_sim (contains all 50 frames of data in mat file), ground_truth (contains the boundary for the data), info_mat (contains info about the sizing and scaling of the images)

def load(im_sim_path, ground_truth_path, info_mat_path):
    sim = sio.loadmat(im_sim_path, squeeze_me = True)
    gt = sio.loadmat(ground_truth_path)
    info = sio.loadmat(info_mat_path, squeeze_me = True)

    im_sim = sim["im_sim"]
    x_gt = gt["X_gt"] 
    aha = gt["aha"] #divides the nodes into segments of heart
    x_axis = info["info"]["x"].item()
    y_axis = info["info"]["y"].item()

    return im_sim, x_gt, aha, x_axis, y_axis

def mm_to_pixels(x_mm, y_mm, x_axis, y_axis): #converts the data in mm to pixel data
    x_px = np.interp(x_mm, x_axis, np.arange(len(x_axis)))
    y_px = np.interp(y_mm, y_axis, np.arange(len(y_axis)))

    return x_px, y_px

def px_to_mm(x_px, y_px, x_axis, y_axis): #converts the pixel data to mm data based on the scaling applied. Useful to obtain displacement
    x_indices = np.arange(len(x_axis))
    y_indices = np.arange(len(y_axis))

    x_mm = np.interp(x_px, x_indices, x_axis)
    y_mm = np.interp(y_px, y_indices, y_axis)

    return (x_mm, y_mm)
def get_contour_pixels(X_gt, x_axis, y_axis, frame_idx): #gets the pixel values of the nodes but closes the boundary
    x_mm = X_gt[:, 0, frame_idx]
    y_mm = X_gt[:, 2, frame_idx]
    x_px, y_px = mm_to_pixels(x_mm, y_mm, x_axis, y_axis)
    return (np.append(x_px, x_px[0]),
            np.append(y_px, y_px[0]))

def get_nodes(x_gt, x_axis, y_axis, frame_index): #gets the pixel values of the nodes but does not close the boundary
    x_mm = x_gt[:, 0, frame_index]
    y_mm = x_gt[:, 2, frame_index]
    return mm_to_pixels(x_mm, y_mm, x_axis, y_axis)

def show_img(img, title = "Echocardiogram", ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize = (8,6))
    else:
        fig = ax.get_figure()
    
    ax.imshow(img, cmap = 'gray', aspect = 'equal')
    ax.set_title(title, fontsize = 12)
    ax.axis("off")
    return fig, ax

def overlay_bound(img, x, y, title = "Ground Truth", line_color = "red", line_width = 2, show_nodes = False, ax = None): #overlay the boundary on the image
    fig, ax = show_img(img, title=title, ax = ax)
    ax.plot(x, y, color = line_color, linewidth = line_width, label = "Ground Truth")

    if show_nodes:
        ax.scatter(x[:-1], y[:-1], s=8, color = "yellow", zorder = 5, label = "nodes")

    ax.legend(loc = "upper right", fontsize = 9)
    return fig, ax




"""
k, l, m, n, o, p =load(ground_truth_path="C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\ground_truth_philips.mat",
       im_sim_path="C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\Data\\solution\\Philips\\im_sim.mat",
       info_mat_path="C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\info.mat")
"""
#print(p)





