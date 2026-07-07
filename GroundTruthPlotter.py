import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch

# 1. Load data : dji41
mat  = sio.loadmat("C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\ground_truth_philips.mat")   # change path if needed
X_gt = mat["X_gt"]                       # shape: (180, 3, 50)
aha  = mat["aha"]                         # shape: (36, 5)

n_points, _, n_frames = X_gt.shape
angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

def get_contours(frame_idx):
    """
    Fixed version for X_gt with axis 1 size = 3
    """
    # Channel 0 = X coordinates
    # Channel 1 = Y coordinates
    # Channel 2 = Likely the 'r' or 'type' identifier (ignored here for normal coords)
    
    x_raw = X_gt[:, 0, frame_idx]
    y_raw = X_gt[:, 2, frame_idx]
    
    # If the dataset only provides one set of (x,y) per frame in this view:
    endo_x = np.append(x_raw, x_raw[0])
    endo_y = np.append(y_raw, y_raw[0])
    
    # We use the same for epi if they are overlapping or if epi is not present
    # Or, if epi is stored in a different variable, you would pull it from there.
    return (endo_x, endo_y)

# 2. Static plot: grid of selected frames
selected_frames = [0, 10, 20, 30, 40, 49]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Speckle Echocardiogram – Ground Truth Contours\n"
             "(Endocardium = red  |  Epicardium = blue)", fontsize=13)

for ax, fi in zip(axes.flat, selected_frames):
    (ex, ey) = get_contours(fi)
    #ax.fill(px, py, color="steelblue", alpha=0.25)
    ax.fill(ex, ey, color="salmon",    alpha=0.35)
    #ax.plot(px, py, color="steelblue", linewidth=1.8, label="Epicardium")
    ax.plot(ex, ey, color="crimson",   linewidth=1.8, label="Endocardium")
    ax.set_title(f"Frame {fi + 1}", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

legend_elements = [Patch(facecolor="crimson",   alpha=0.6, label="Endocardium")]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("contours_grid.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: contours_grid.png")



fig2, ax2 = plt.subplots(figsize=(10, 4))
#ax2.plot(range(1, n_frames + 1), mean_endo, color="crimson",   lw=2, label="Mean endo radius")
#ax2.plot(range(1, n_frames + 1), mean_epi,  color="steelblue", lw=2, label="Mean epi radius")
#ax2.fill_between(range(1, n_frames + 1), mean_endo, mean_epi,
                 #alpha=0.15, color="gray", label="Wall thickness region")
ax2.set_xlabel("Frame", fontsize=11)
ax2.set_ylabel("Radius (pixels)", fontsize=11)
ax2.set_title("Mean Contour Radius Over Cardiac Cycle (all 50 frames)", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("radii_over_time.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: radii_over_time.png")

