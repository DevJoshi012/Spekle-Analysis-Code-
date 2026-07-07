#create windows along the nodes 

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from matplotlib.patches import Patch
import matplotlib.patches as mpatches


"""
Two functions:
  1. get_window_coords  - given node pixel coordinates, return the window
                          corner coordinates for each node
  2. extract_and_display - slice image data from im_sim at each window
                           and display the patches
"""




# ── Function 1 ────────────────────────────────────────────────────────────────

def get_window_coords(node_px_x, node_px_y, half_size=10):
    """
    Given node pixel coordinates, return the top-left and bottom-right
    corner of the window box for each node.

    Parameters
    ----------
    node_px_x : np.ndarray (n_nodes,)  x pixel coords of each node
    node_px_y : np.ndarray (n_nodes,)  y pixel coords of each node
    half_size : int  half width/height of the window (default 10 → 21x21)

    Returns
    -------
    windows : list of dicts, one per node, each with keys:
                'node'    : int    node index
                'cx'      : float  centre x in pixels
                'cy'      : float  centre y in pixels
                'r0','r1' : int    row    start/end in image
                'c0','c1' : int    column start/end in image
    """
    windows = []
    
    for i, (cx, cy) in enumerate(zip(node_px_x, node_px_y)):
        r0 = int(round(cy)) - half_size
        r1 = int(round(cy)) + half_size + 1
        c0 = int(round(cx)) - half_size
        c1 = int(round(cx)) + half_size + 1
        windows.append(dict(node=i, cx=cx, cy=cy,
                            r0=r0, r1=r1, c0=c0, c1=c1))
    return windows


# ── Function 2 ────────────────────────────────────────────────────────────────

def extract_and_display(im_sim, windows, frame_idx,
                         node_indices=None, save_path=None):
    """
    Slice image data from im_sim at each window location and display.
    Left panel  : full frame with window boxes drawn on the boundary.
    Right panel : extracted patch for each selected node.

    Parameters
    ----------
    im_sim       : np.ndarray (H, W, n_frames)
    windows      : list of dicts from get_window_coords()
    frame_idx    : int  which frame to use (zero-based)
    node_indices : list of int or None  which nodes to show
                   (None = every 15th node)
    save_path    : str or None

    Returns
    -------
    fig
    """
    if node_indices is None:
        node_indices = list(range(0, len(windows), 15))

    img    = im_sim[:, :, frame_idx]
    H, W   = img.shape
    n_show = len(node_indices)
    colors = plt.cm.tab20(np.linspace(0, 1, n_show))

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(f"Node Windows — Frame {frame_idx + 1}", fontsize=13,
                 fontweight="bold")

    # ── Left panel: full image + window boxes ────────────────────────────────
    ax = fig.add_axes([0.01, 0.05, 0.44, 0.88])
    ax.imshow(img, cmap="gray", aspect="equal")

    # draw all nodes as small yellow dots
    all_cx = [w["cx"] for w in windows]
    all_cy = [w["cy"] for w in windows]
    ax.scatter(all_cx, all_cy, s=6, color="yellow", alpha=0.6, zorder=4)

    # draw coloured window box for each selected node
    for col, ni in zip(colors, node_indices):
        w    = windows[ni]
        size = w["r1"] - w["r0"]
        rect = mpatches.Rectangle(
            (w["c0"], w["r0"]), size, size,
            linewidth=1.2, edgecolor=col, facecolor="none", zorder=5
        )
        ax.add_patch(rect)
        ax.scatter(w["cx"], w["cy"], s=40, color=col, zorder=6)
        ax.text(w["c1"] + 2, w["cy"], str(ni),
                color=col, fontsize=6, va="center")

    ax.set_title(f"Frame {frame_idx + 1} — All Nodes + Selected Windows",
                 fontsize=10)
    ax.axis("off")

    # ── Right panel: extracted patches ───────────────────────────────────────
    cols_p = 4
    rows_p = int(np.ceil(n_show / cols_p))

    for idx, ni in enumerate(node_indices):
        w = windows[ni]

        # clamp to image bounds before slicing
        r0 = max(0, w["r0"]);  r1 = min(H, w["r1"])
        c0 = max(0, w["c0"]);  c1 = min(W, w["c1"])
        patch = img[r0:r1, c0:c1]

        r = idx // cols_p
        c = idx  % cols_p
        left   = 0.48 + c * 0.13
        bottom = 0.08 + (rows_p - 1 - r) * (0.82 / rows_p)
        ax_p   = fig.add_axes([left, bottom, 0.11, 0.75 / rows_p])
        ax_p.imshow(patch, cmap="gray", vmin=0, vmax=255)
        ax_p.set_title(f"Node {ni}", fontsize=7,
                       color=colors[idx], fontweight="bold")
        ax_p.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig

def extract(im_sim, windows, frame_idx,
                         node_idx):
    #method to return extracted data of a node / search area as per definition
    #windows already handles half size (size of the window) parameter so specifying the search area size / variable sizing must be done upstream of the function

    H, W = im_sim.shape[0], im_sim.shape[1]

    w = windows[node_idx] #gets the window dictionary related to node index provided

    r0 = max(0, w['r0'])
    c0 = max(0, w['c0'])
    r1 = min(H, w['r1'])
    c1 = min(W, w['c1'])

    return im_sim[r0:r1, c0:c1, frame_idx] #returns the required matrix