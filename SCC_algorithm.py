#steps: set up a basic NCC algorithm, slide over the target window over the search window: use FFT for faster results

#imports
import numpy as np
from scipy.signal import correlate2d #performs cross-correlation between two matrices using FFT as its basis algorithm
from scipy.optimize import curve_fit #least squares non-linear curve fitting module
from scipy.ndimage import uniform_filter

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from boundary_util import load, get_nodes, mm_to_pixels, px_to_mm #in-house tool which loads all the boundaries from ground truth and gets all the respective nodes from the boundaries drawm
from window_nodes import get_window_coords, extract #in-house tool which gets the coordinates for the target windows as we would call them and we shall use these windows to create our search areas

#importing student gaussian apodization file:
from student_gaussian_apodization_peak_finding import gaussian_apodization_window, find_correlation_peak_2d

def normalized_cross_correlation(template, search): #template window is the smaller window and search window is the larger window
    t = template.astype(np.float64) #convert the template window (0,255) pixel values to float values since they are more mathematically accurate for getting lower errors after doing fft convolution on them
    s = search.astype(np.float64) #same setup but for the search area

    t -= t.mean() #subtract the mean value from all the pixels to achieve "light invariance", removes the inherently higher mathematical score advantage of brighter pixels

    t_norm = np.linalg.norm(t) #normalize the matrix t
    if t_norm < 1e-6:
        return np.zeros((s.shape[0] - t.shape[0] + 1, s.shape[1] - t.shape[1] + 1)) #basically returns the array of zeros of the height and width of the template size if the std dev is less than tolerance
    

    t /= t_norm
    raw = correlate2d(s, t, mode="valid") #use fft to correlate t and s matrices

    th, tw = t.shape
    sh, sw = s.shape
    oh, ow = sh - th + 1, sw - tw + 1
    ncc    = np.zeros((oh, ow))

    for r in range(oh):
        for c in range(ow):
            patch      = s[r:r+th, c:c+tw]
            patch      = patch - patch.mean()
            n          = np.linalg.norm(patch)
            ncc[r, c]  = raw[r, c] / (n + 1e-8)

    return ncc

#scc correlation algorithm which is friendly to gaussian apodization
def scc_correlation(template, search):
    """
    FFT-based correlation plane. Apodization friendly unlike ncc
    """
    t = template.astype(np.float64)
    s = search.astype(np.float64)

    # mean-subtract both (removes DC / brightness offset)
    t -= t.mean()
    s -= s.mean()

    # pad template to search size so FFTs are compatible
    th, tw = t.shape
    sh, sw = s.shape
    t_padded = np.zeros_like(s)
    t_padded[:th, :tw] = t

    # FFT correlation: conj(F(t)) * F(s), inverse transform
    F_t = np.fft.fft2(t_padded)
    F_s = np.fft.fft2(s)
    corr = np.fft.ifft2(np.conj(F_t) * F_s).real

    # shift zero-lag to the centre so the peak finder's centring works
    corr = np.fft.fftshift(corr)
    return corr

#scc correlation with normalization:
def scc_correlation_normalized(template, search, apodization_window=None):
    """
    Normalized FFT cross-correlation plane.
 
    Parameters
    ----------
    template          : 2D array (th, tw)
    search            : 2D array (sh, sw), must be larger than template
    apodization_window: optional 2D array, same shape as template.
                        NOTE: on this data apodization did NOT help even after
                        normalization (it degrades the match). Left optional
                        for experimentation only.
 
    Returns
    -------
    ncc : 2D normalized correlation plane, zero-lag at the centre.
    """
    t = template.astype(np.float64)
    s = search.astype(np.float64)
 
    if apodization_window is not None:
        t = t * apodization_window
 
    t -= t.mean()
 
    th, tw = t.shape
    sh, sw = s.shape
 
    # --- numerator: FFT correlation of mean-subtracted template with search ---
    t_padded = np.zeros_like(s)
    r0 = (sh - th) // 2
    c0 = (sw - tw) // 2
    t_padded[r0:r0 + th, c0:c0 + tw] = t
 
    num = np.fft.fftshift(
        np.fft.ifft2(np.conj(np.fft.fft2(t_padded)) * np.fft.fft2(s)).real
    )
 
    # --- denominator: local energy of the search at every window position ---
    # local std over a (th x tw) window = sqrt(E[x^2] - E[x]^2)
    win_mean   = uniform_filter(s,     size=(th, tw), mode="constant")
    win_sqmean = uniform_filter(s * s, size=(th, tw), mode="constant")
    win_var    = np.maximum(win_sqmean - win_mean ** 2, 0.0)
    win_norm   = np.sqrt(win_var) * np.sqrt(th * tw)
 
    t_norm = np.linalg.norm(t)
    denom  = t_norm * win_norm + 1e-8
 
    return num / denom
 

#using the above function to cross correlate the known synthetic data
t_half_size = 10 #template half size
s_half_size = 40 #search half size

#required paths
im_sim_path = "C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\Data\\solution\\Philips\\im_sim.mat"
ground_truth_path = "C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\ground_truth_philips.mat"
info_mat_path = "C:\\Users\\Dev Joshi\\Box\\aether.lab\\people\\Dev Joshi\\SpekleAnalysis\\GroundTruthOpen\\info.mat"

im_sim, x_gt, aha, x_axis, y_axis = load(im_sim_path, ground_truth_path, info_mat_path)




#loop implementation for all nodes scc function:
def single_scc(t_half_size, s_half_size, frame=49, window_size=0.5):
    x_t, y_t = get_nodes(x_gt, x_axis, y_axis, frame-1)
    temp_windows   = get_window_coords(x_t, y_t, t_half_size)
    search_windows = get_window_coords(x_t, y_t, s_half_size)

    # apodization window sized to the TEMPLATE
    gw = gaussian_apodization_window(
        dimensions=(2*t_half_size+1, 2*t_half_size+1),
        window_size=(window_size, window_size),
        window_type='fraction'
    )

    out = []
    for i in range(len(temp_windows)):
        t = extract(im_sim, temp_windows,   frame-1, i).astype(np.float64) #frame - 1 changed to frame 
        s = extract(im_sim, search_windows, frame,   i).astype(np.float64)

        #t_windowed = t * gw                    # apodize BEFORE the FFT
        t_windowed = t
        corr = scc_correlation_normalized(t_windowed, s)  # FFT correlation plane

        peak = find_correlation_peak_2d(corr)  # designed for this plane
        out.append((peak['u'], peak['v']))

    return out



"""
All functions below this are using NCC instead of SCC implementation in correlation, hence do not result good results implementing apodization
"""
#function to iteratively get all nodal displacements based on the frame given: we have 50 frames in total

def single_ncc(t_half_size, s_half_size, frame=49):

    #image1 = im_sim[:, :, frame - 1]

    x_t, y_t = get_nodes(x_gt, x_axis, y_axis, frame-1)
    x_s, y_s = get_nodes(x_gt, x_axis, y_axis, frame)

    # get window corner coords for all nodes
    temp_windows = get_window_coords(x_t, y_t, t_half_size)
    search_windows = get_window_coords(x_t, y_t, s_half_size)

    #for all the nodal indices, make 2 2D lists of nodal slice data which correspond to template and search regions
    template_data = []
    search_data = []

    for i in range(len(temp_windows)): #since temp and search windows have same dimensions
        t = extract(im_sim, temp_windows, frame - 1, i)
        s = extract(im_sim, search_windows, frame, i)

        template_data.append(t)
        search_data.append(s)

    node_disp_output = []
    #for all the available nodes:
    for i in range(len(temp_windows)):
        #compute displacement distance:
        ncc_result = normalized_cross_correlation(template_data[i], search_data[i])

        pr, pc = np.unravel_index(np.argmax(ncc_result), ncc_result.shape) #pixel location of the highest/ peak value of correlation found by ncc algorithm
        dy = pr - (s_half_size - t_half_size)
        
        dx = pc - (s_half_size - t_half_size)

        #p, q = px_to_mm(dx, dy, x_axis, y_axis)
        #pn, qn = float(p), float(q) #convert to regular float numbers from numpy float 64 numbers

        node_disp_output.append((dx, dy))


    return node_disp_output

#function to use only find peak correlation function from the provided code to perform cross correlation:
def single_ncc_improved(t_half_size, s_half_size, frame=49):
    x_t, y_t = get_nodes(x_gt, x_axis, y_axis, frame-1)
    
    temp_windows = get_window_coords(x_t, y_t, t_half_size)
    search_windows = get_window_coords(x_t, y_t, s_half_size)
    
    template_data = []
    search_data = []

    for i in range(len(temp_windows)):
        t = extract(im_sim, temp_windows, frame - 1, i)
        s = extract(im_sim, search_windows, frame, i)

        template_data.append(t)
        search_data.append(s)

    node_disp_output = []
    for i in range(len(temp_windows)):
        ncc_result = normalized_cross_correlation(template_data[i], search_data[i])
        
        # Use subpixel peak finding instead of argmax
        peak_info = find_correlation_peak_2d(ncc_result)
        
        # Extract displacement from peak_info
        dx = peak_info['u']
        dy = peak_info['v']
        
        node_disp_output.append((dx, dy))

    return node_disp_output

#guassian apodization included ncc
def single_ncc_gaussian(t_half_size, s_half_size, frame=49):
    x_t, y_t = get_nodes(x_gt, x_axis, y_axis, frame-1)
    
    temp_windows = get_window_coords(x_t, y_t, t_half_size)
    search_windows = get_window_coords(x_t, y_t, s_half_size)
    
    # Create a SOFT Gaussian window (window_size=0.9 means it spans 90% of patch)
    gauss_window = gaussian_apodization_window(
        dimensions=(2*t_half_size+1, 2*t_half_size+1),
        window_size=(0.9, 0.9),
        window_type='fraction'
    )
    
    template_data = []
    search_data = []

    for i in range(len(temp_windows)):
        t = extract(im_sim, temp_windows, frame - 1, i)
        s = extract(im_sim, search_windows, frame, i)
        
        # Apply window to template ONLY
        t_windowed = t.astype(np.float64) * gauss_window
        
        template_data.append(t_windowed)
        search_data.append(s)

    node_disp_output = []
    for i in range(len(template_data)):
        ncc_result = normalized_cross_correlation(template_data[i], search_data[i])
        pr, pc = np.unravel_index(np.argmax(ncc_result), ncc_result.shape)
        dy = pr - (s_half_size - t_half_size)
        dx = pc - (s_half_size - t_half_size)
        node_disp_output.append((dx, dy))

    return node_disp_output

#making a new gaussian function
def single_ncc_gaussian_new(t_half_size, s_half_size, frame=49):
    x_t, y_t = get_nodes(x_gt, x_axis, y_axis, frame-1)
    
    temp_windows = get_window_coords(x_t, y_t, t_half_size)
    search_windows = get_window_coords(x_t, y_t, s_half_size)
    
    gauss_window = gaussian_apodization_window(
        dimensions=(2 * t_half_size + 1, 2 * t_half_size + 1),
        window_size=(0.5, 0.5),
        window_type='fraction'
    )
    
    template_data = []
    search_data = []

    for i in range(len(temp_windows)):
        t = extract(im_sim, temp_windows, frame - 1, i)
        s = extract(im_sim, search_windows, frame, i)
        t_windowed = t.astype(np.float64) * gauss_window
        template_data.append(t_windowed)
        search_data.append(s)

    node_disp_output = []
    for i in range(len(template_data)):
        ncc_result = normalized_cross_correlation(template_data[i], search_data[i])
        peak_info = find_correlation_peak_2d(ncc_result)
        
        dx = peak_info['u']
        dy = peak_info['v']
        
        node_disp_output.append((dx, dy))
    
    return node_disp_output
        

    #return node_disp_output


"""
Utility functions : to plot errors, confidence maps, displacement vectors as well as generate animation gifs
"""

#plotting the displacement vectors of the provided data
def plot_displacement_vectors(im_sim, x1, y1, x2, y2, arr_pred, frame_idx=49, arrow_scale=20):
    """
    Parameters
    ----------
    arrow_scale : int  multiply displacements by this to make arrows visible
                       (purely visual, does not affect values)
    """
    img = im_sim[:, :, frame_idx]

    gt_dx = (x2 - x1) * arrow_scale
    gt_dy = (y2 - y1) * arrow_scale
    pred_dx = arr_pred[:, 0] * arrow_scale
    pred_dy = arr_pred[:, 1] * arrow_scale

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap='gray', aspect='equal')

    # ground truth vectors in green
    ax.quiver(x1, y1, gt_dx, gt_dy,
              color='lime',
              angles='xy', scale_units='xy', scale=1,
              width=0.003, headwidth=4, headlength=4,
              label='Ground Truth')

    # predicted vectors in red
    ax.quiver(x1, y1, pred_dx, pred_dy,
              color='red',
              angles='xy', scale_units='xy', scale=1,
              width=0.003, headwidth=4, headlength=4,
              label='Predicted')

    # also plot node positions so you can see them even if arrows overlap
    ax.scatter(x1, y1, s=8, color='yellow', zorder=5)

    ax.legend(loc='upper right', fontsize=10,
              facecolor='black', labelcolor='white')
    ax.set_title(f'Displacement Vectors — Frame {frame_idx} → {frame_idx + 1}  '
                 f'(arrows scaled ×{arrow_scale})\n'
                 f'Green = Ground Truth   Red = Predicted', fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

#create animation of all 50 frames:
def animate_displacement_vectors(im_sim, x_gt, x_axis, y_axis, 
                                  single_ncc_func, t_half_size, s_half_size,
                                  mm_to_pixels_func, name = 'displacement_animation', arrow_scale=20, fps=10):
    """
    Create an animated sequence of displacement vectors for all frame pairs.
    
    Parameters
    ----------
    single_ncc_func : callable  your single_ncc function
    mm_to_pixels_func : callable  your mm_to_pixels function
    fps : int  frames per second for animation
    """
    import matplotlib.animation as animation
    
    n_frames = im_sim.shape[2]
    #n_frames = 3
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame_num):
        ax.clear()
        
        if frame_num >= n_frames - 1:
            return []
        
        # Get nodes for this frame pair
        x1, y1 = mm_to_pixels_func(x_gt[:, 0, frame_num], 
                                    x_gt[:, 2, frame_num], 
                                    x_axis, y_axis)
        x2, y2 = mm_to_pixels_func(x_gt[:, 0, frame_num + 1], 
                                    x_gt[:, 2, frame_num + 1], 
                                    x_axis, y_axis)
        
        # Compute predictions
        arr_pred = np.array(single_ncc_func(t_half_size, s_half_size, frame=frame_num + 1))
        
        # Ground truth displacement
        gt_dx = (x2 - x1) * arrow_scale
        gt_dy = (y2 - y1) * arrow_scale
        pred_dx = arr_pred[:, 0] * arrow_scale
        pred_dy = arr_pred[:, 1] * arrow_scale
        
        # Display
        img = im_sim[:, :, frame_num]
        ax.imshow(img, cmap='gray', aspect='equal')
        
        ax.quiver(x1, y1, gt_dx, gt_dy,
                  color='lime', angles='xy', scale_units='xy', scale=1,
                  width=0.003, headwidth=4, headlength=4,
                  label='Ground Truth')
        
        ax.quiver(x1, y1, pred_dx, pred_dy,
                  color='red', angles='xy', scale_units='xy', scale=1,
                  width=0.003, headwidth=4, headlength=4,
                  label='Predicted')
        
        ax.scatter(x1, y1, s=8, color='yellow', zorder=5)
        
        ax.legend(loc='upper right', fontsize=10,
                  facecolor='black', labelcolor='white')
        ax.set_title(f'Displacement Vectors — Frame {frame_num + 1} → {frame_num + 2}  '
                     f'(arrows scaled ×{arrow_scale})',
                     fontsize=11)
        ax.axis('off')
        
        print(f"{frame_num} done")
        return []
    
    anim = animation.FuncAnimation(fig, update, frames=n_frames - 1,
                                    interval=1000/fps, blit=False)
    
    # Save to file
    anim.save(f'{name}.gif', writer='pillow', fps=fps)
    print(f'Saved: {name}.gif')
    plt.show()

#plotting the confidence map using peak and deviation from the peak
def plot_ncc_confidence_map(im_sim, x_gt, x_axis, y_axis, 
                             single_ncc_func, t_half_size, s_half_size,
                             get_nodes_func, mm_to_pixels_func,
                             frame_idx=1, save_path=None):
    """
    Colour-code each node by its NCC peak value (confidence).
    Green/blue = high confidence (peak close to 1.0)
    Red/yellow = low confidence (peak close to 0)
    
    Parameters
    ----------
    single_ncc_func : callable  your single_ncc function (returns displacements)
    frame_idx : int  which frame pair to visualise
    """
    img = im_sim[:, :, frame_idx]
    
    # Get node positions for this frame
    x_nodes, y_nodes = get_nodes_func(x_gt, x_axis, y_axis, frame_idx)
    
    # Compute displacements AND collect NCC peaks for each node
    # We need to modify single_ncc to also return peak values
    # For now, let's compute them inline:
    
    from window_nodes import get_window_coords, extract
    
    x_t, y_t = get_nodes_func(x_gt, x_axis, y_axis, frame_idx)
    temp_windows = get_window_coords(x_t, y_t, t_half_size)
    search_windows = get_window_coords(x_t, y_t, s_half_size)
    
    # Recompute NCC for each node to capture peak values
    peak_values = []
    for i in range(len(temp_windows)):
        t = extract(im_sim, temp_windows, frame_idx, i)
        s = extract(im_sim, search_windows, frame_idx + 1, i)
        
        ncc_result = normalized_cross_correlation(t, s)
        peak_value = ncc_result.max()
        peak_values.append(peak_value)
    
    peak_values = np.array(peak_values)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap='gray', aspect='equal')
    
    # Scatter plot nodes colour-coded by peak value
    scatter = ax.scatter(x_nodes, y_nodes, c=peak_values, 
                         cmap='RdYlGn', s=50, vmin=0, vmax=1,
                         zorder=5, edgecolors='white', linewidth=0.5)
    
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('NCC Peak Value\n(1.0 = perfect match, 0 = no match)', 
                   fontsize=10)
    
    ax.set_title(f'NCC Confidence Map — Frame {frame_idx} → {frame_idx + 1}\n'
                 f'Green = High confidence (easy to track)  |  '
                 f'Red = Low confidence (hard to track)',
                 fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    
    plt.show()
    
    # Print statistics
    print(f'\nNCC Peak Value Statistics (Frame {frame_idx} → {frame_idx + 1}):')
    print(f'  Mean:    {peak_values.mean():.4f}')
    print(f'  Median:  {np.median(peak_values):.4f}')
    print(f'  Min:     {peak_values.min():.4f}  (node {peak_values.argmin()})')
    print(f'  Max:     {peak_values.max():.4f}  (node {peak_values.argmax()})')
    print(f'  Std:     {peak_values.std():.4f}')
    print(f'  Nodes with peak < 0.7 (low confidence): {np.sum(peak_values < 0.7)}/180')
    
    return peak_values

#mapping the error across all the frames using specified ncc function
def plot_error_map_all_frames(im_sim, x_gt, x_axis, y_axis,
                               single_ncc_func, t_half_size, s_half_size,
                               get_nodes_func, mm_to_pixels_func,
                               save_path=None):
    """
    Show displacement error (predicted vs ground truth) for all 49 frame pairs.
    Red nodes = high error (bad tracking)
    Green nodes = low error (good tracking)
    """
    n_frames = im_sim.shape[2]
    n_pairs = n_frames - 1
    
    cols = 7
    rows = int(np.ceil(n_pairs / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 14))
    axes = np.array(axes).flatten()
    
    fig.suptitle('Displacement Error Maps — All 49 Frame Pairs\n'
                 'Green = Low Error (accurate)  |  Red = High Error (inaccurate)',
                 fontsize=13, fontweight='bold')
    
    all_error_stats = []
    
    for frame_idx in range(n_pairs):
        ax = axes[frame_idx]
        img = im_sim[:, :, frame_idx]
        
        # Ground truth displacement
        x1, y1 = mm_to_pixels_func(x_gt[:, 0, frame_idx], 
                                    x_gt[:, 2, frame_idx], 
                                    x_axis, y_axis)
        x2, y2 = mm_to_pixels_func(x_gt[:, 0, frame_idx + 1], 
                                    x_gt[:, 2, frame_idx + 1], 
                                    x_axis, y_axis)
        
        gt_dx = x2 - x1
        gt_dy = y2 - y1
        gt_disp = np.array(list(zip(gt_dx, gt_dy)))
        
        # Predicted displacement
        pred_disp = np.array(single_ncc_func(t_half_size, s_half_size, frame=frame_idx + 1))
        
        # Calculate error per node
        differences = pred_disp - gt_disp
        errors = np.sqrt(np.sum(differences**2, axis=1))  # Euclidean distance
        
        all_error_stats.append({
            'frame': frame_idx,
            'mean_error': errors.mean(),
            'max_error': errors.max(),
            'high_error_count': np.sum(errors > 1.0)
        })
        
        # Plot on this subplot
        ax.imshow(img, cmap='gray', aspect='equal')
        scatter = ax.scatter(x1, y1, c=errors,
                             cmap='RdYlGn_r', s=30, vmin=0, vmax=2,
                             zorder=5, edgecolors='none')
        
        ax.set_title(f'F{frame_idx+1}→{frame_idx+2}  mean_err={errors.mean():.3f}px',
                     fontsize=8)
        ax.axis('off')
        print(f"{frame_idx} / {n_frames} done")
    # Hide unused subplots
    for j in range(frame_idx + 1, len(axes)):
        axes[j].axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                                norm=plt.Normalize(vmin=0, vmax=2))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Displacement Error (pixels)', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    
    plt.show()
    
    # Print summary statistics
    print('\n=== Displacement Error Summary (All Frames) ===')
    print(f'{"Frame Pair":<12} {"Mean Error":<12} {"Max Error":<12} {"Nodes > 1px":<15}')
    print('-' * 51)
    for stat in all_error_stats:
        print(f"F{stat['frame']+1}→{stat['frame']+2:<8} {stat['mean_error']:.4f}px       "
              f"{stat['max_error']:.4f}px       {stat['high_error_count']}/180")
    
    mean_error_all = np.mean([s['mean_error'] for s in all_error_stats])
    print('-' * 51)
    print(f'Overall mean error: {mean_error_all:.4f} pixels')
    print(f'Frames with high error (mean > 1.5px): '
          f'{sum(1 for s in all_error_stats if s["mean_error"] > 1.5)}/49')
    
    return all_error_stats

#frame 1 to 2 displacements actual
x1, y1 = mm_to_pixels(x_gt[:, 0, 1], x_gt[:, 2, 1], x_axis, y_axis)
x2, y2 = mm_to_pixels(x_gt[:, 0, 2], x_gt[:, 2, 2], x_axis, y_axis)


dxg = x2-x1
dyg = y2-y1

def evaluate_tracking_performance(arr_gt, arr_pred):
    """
    Calculates and prints tracking performance metrics:
    - Mean Displacement Error (MDE)
    - Dimensionless Tracking Accuracy (Correlation)
    - Normalized Root Mean Squared Error (NRMSE)
    """
    # 1. Compute the straight-line (Euclidean) distance error for each displacement pair
    differences = arr_pred - arr_gt
    errors = np.sqrt(np.sum(differences**2, axis=1))
    mean_distance_error = np.mean(errors).item()

    # Flatten the arrays to evaluate the whole system globally
    gt_flat = arr_gt.flatten()
    pred_flat = arr_pred.flatten()

    # 2. Calculate the correlation coefficient matrix safely
    correlation_matrix = np.corrcoef(gt_flat, pred_flat)
    # Handle potential nan if data has zero variance
    tracking_accuracy = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0

    # 3. Calculate the raw Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((gt_flat - pred_flat) ** 2))

    # 4. Calculate the peak-to-peak range of your ground truth
    gt_range = np.ptp(gt_flat)

    # Prevent division by zero if there is absolutely zero movement
    if gt_range == 0:
        nrmse = 0.0 if rmse == 0 else float('inf')
    else:
        nrmse = rmse / gt_range

    # Print the formatted metrics
    print("------- Tracking Evaluation Metrics -------")
    print(f"Mean Displacement Error:       {mean_distance_error:.4f} px")
    print(f"Dimensionless Tracking Accuracy: {tracking_accuracy:.4f}")
    print(f"Normalized Error (NRMSE):       {nrmse:.4f}")
    print("-------------------------------------------")
#displacement visualization for one (specified or default frame)
#plot_displacement_vectors(im_sim, x1, y1, x2, y2, arr_pred)

"""
implementing the functions to generate results here:
"""

"""
peak_values = plot_ncc_confidence_map(im_sim, x_gt, x_axis, y_axis,
                                      single_ncc, t_half_size, s_half_size,
                                      get_nodes, mm_to_pixels,
                                      frame_idx=1)
"""
"""
animate_displacement_vectors(im_sim, x_gt, x_axis, y_axis,
                              single_ncc_gaussian, t_half_size, s_half_size,
                              mm_to_pixels, name = 'ncc_gaussian_new', arrow_scale=20, fps=5)
"""


animate_displacement_vectors(im_sim, x_gt, x_axis, y_axis,
                              single_scc, t_half_size, s_half_size,
                              mm_to_pixels, name='scc_gaussian_2', arrow_scale=20, fps=5)

"""
error_stats2 = plot_error_map_all_frames(im_sim, x_gt, x_axis, y_axis,
                                        single_scc, t_half_size, s_half_size,
                                        get_nodes, mm_to_pixels,
                                        save_path='error_map_all_frames_scc.png')

"""
