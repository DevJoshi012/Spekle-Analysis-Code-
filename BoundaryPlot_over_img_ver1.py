import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ── Paths (update these to your local paths) ────────────────────────────────
IMAGE_PATH       = "C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\image_001.tiff"
GT_PATH          = "C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\ground_truth_philips.mat"
INFO_PATH        = "C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\info.mat"


# ── Load data ────────────────────────────────────────────────────────────────
def load_data(image_path, gt_path, info_path):
    """
    Load the echocardiogram image, ground truth contours, and axis info.

    Returns
    -------
    img    : np.ndarray  (H, W) uint8 grayscale image
    X_gt   : np.ndarray  (180, 3, 50) ground truth contour coordinates in mm
    x_axis : np.ndarray  (W,) physical x-axis values in mm, one per pixel column
    y_axis : np.ndarray  (H,) physical y-axis values in mm, one per pixel row
    """
    img    = np.array(Image.open(image_path))
    gt     = sio.loadmat(gt_path)
    X_gt   = gt["X_gt"]
    info   = sio.loadmat(info_path, squeeze_me=True)
    x_axis = info["info"]["x"].item()
    y_axis = info["info"]["y"].item()
    return img, X_gt, x_axis, y_axis


# ── Convert mm coordinates to pixel coordinates ──────────────────────────────
def mm_to_pixels(x_mm, y_mm, x_axis, y_axis):
    """
    Convert contour coordinates from millimetres to pixel indices using
    the physical axis vectors stored in info.mat.

    Parameters
    ----------
    x_mm   : np.ndarray  x coordinates in mm
    y_mm   : np.ndarray  y coordinates in mm
    x_axis : np.ndarray  mm value for each image column (from info.mat)
    y_axis : np.ndarray  mm value for each image row    (from info.mat)

    Returns
    -------
    x_px, y_px : np.ndarray pixel coordinates (floats)
    """
    x_px = np.interp(x_mm, x_axis, np.arange(len(x_axis)))
    y_px = np.interp(y_mm, y_axis, np.arange(len(y_axis)))
    return x_px, y_px


# ── Get closed contour for a given frame ────────────────────────────────────
def get_contour_pixels(X_gt, x_axis, y_axis, frame_idx):
    """
    Extract the boundary contour for a single frame and return it in
    pixel coordinates, closed (first point appended at end).

    Parameters
    ----------
    X_gt      : np.ndarray  (180, 3, n_frames) ground truth array
    x_axis    : np.ndarray  mm axis for columns
    y_axis    : np.ndarray  mm axis for rows
    frame_idx : int         zero-based frame index

    Returns
    -------
    x_px, y_px : np.ndarray closed contour in pixel coordinates
    """
    x_mm = X_gt[:, 0, frame_idx]
    y_mm = X_gt[:, 2, frame_idx]
    x_px, y_px = mm_to_pixels(x_mm, y_mm, x_axis, y_axis)

    # Close the contour by appending the first point
    x_px = np.append(x_px, x_px[0])
    y_px = np.append(y_px, y_px[0])
    return x_px, y_px


# ── Open and display the image with matplotlib ───────────────────────────────
def show_image(img, title="Echocardiogram", ax=None):
    """
    Display a grayscale echocardiogram image using matplotlib.

    Parameters
    ----------
    img   : np.ndarray  (H, W) grayscale image
    title : str         plot title
    ax    : matplotlib Axes, optional. Creates a new figure if None.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    ax.imshow(img, cmap="gray", aspect="equal")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    return fig, ax


# ── Overlay the boundary contour on the image ────────────────────────────────
def overlay_boundary(img, x_px, y_px, title="Frame 1 – Ground Truth Boundary",
                     line_color="red", line_width=2, ax=None):
    """
    Overlay a ground truth boundary contour on an echocardiogram image.

    Parameters
    ----------
    img        : np.ndarray  (H, W) grayscale image
    x_px       : np.ndarray  contour x pixel coordinates (closed)
    y_px       : np.ndarray  contour y pixel coordinates (closed)
    title      : str         plot title
    line_color : str         contour line colour (default 'red')
    line_width : float       contour line width  (default 2)
    ax         : matplotlib Axes, optional. Creates a new figure if None.

    Returns
    -------
    fig, ax
    """
    fig, ax = show_image(img, title=title, ax=ax)
    ax.plot(x_px, y_px, color=line_color, linewidth=line_width,
            label="GT Boundary")
    ax.legend(loc="upper right", fontsize=10)
    return fig, ax


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load everything
    img, X_gt, x_axis, y_axis = load_data(IMAGE_PATH, GT_PATH, INFO_PATH)

    # Get contour for frame 1 (index 0)
    x_px, y_px = get_contour_pixels(X_gt, x_axis, y_axis, frame_idx=0)

    # Plot overlay
    fig, ax = overlay_boundary(img, x_px, y_px,
                               title="Frame 1 – Ground Truth Boundary")
    plt.tight_layout()
    plt.savefig("overlay_frame1.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: overlay_frame1.png")