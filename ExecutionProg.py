#example execution:

#import both modules

import boundary_util as bu
import window_nodes as wn

#required paths
im_sim_path = "C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\Data\\solution\\Philips\\im_sim.mat"
ground_truth_path = "C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\ground_truth_philips.mat"
info_mat_path = "C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\info.mat"


im_sim, x_gt, aha, x_axis, y_axis = bu.load(im_sim_path, ground_truth_path, info_mat_path)

frame = 2 #load frame number start from 1 to 50

image = im_sim[:, :, frame - 1]

x, y = bu.get_nodes(x_gt, x_axis, y_axis, frame)


fig, ax = bu.overlay_bound(image, x, y, title=f"Ground truth for {frame - 1} frame", show_nodes=True)

    # get window corner coords for all nodes
windows = wn.get_window_coords(x, y, half_size=10)

    # display
fig = wn.extract_and_display(im_sim, windows, frame_idx=0)

bu.plt.show()