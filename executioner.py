import scc_module as sm

#run all the tests here to keep the scc_module file clean and organized with helper and main functions. 


#All data loading steps have been implemented here to load relevant data for the scc_module file.
t_half_size = 10 #template half size
s_half_size = 20 #search half size

#gaussian mega study:
scc_gaussian_animation = sm.animate_displacement_vectors(sm.im_sim, sm.x_gt, sm.x_axis, sm.y_axis, sm.single_scc_weighted, t_half_size, s_half_size, sm.mm_to_pixels, name='scc_gaussian', arrow_scale=20, fps=5)
scc_gaussian_errorplot = sm.plot_error_map_all_frames_csv(sm.im_sim, sm.x_gt, sm.x_axis, sm.y_axis, sm.single_scc_weighted, t_half_size, s_half_size, sm.mm_to_pixels, save_path='scc_gaussian_error.png', csv_summary_path='scc_gaussian_summary.csv', csv_nodes_path='scc_gaussian_nodes.csv')

#non gaussian mega study:
non_scc_gaussian_animation = sm.animate_displacement_vectors(sm.im_sim, sm.x_gt, sm.x_axis, sm.y_axis, sm.single_scc_nongauss, t_half_size, s_half_size, sm.mm_to_pixels, name='scc_nongaussian', arrow_scale=20, fps=5)
non_scc_gaussian_errorplot = sm.plot_error_map_all_frames_csv(sm.im_sim, sm.x_gt, sm.x_axis, sm.y_axis, sm.single_scc_nongauss, t_half_size, s_half_size, sm.mm_to_pixels, save_path='scc_nongaussian_error.png', csv_summary_path='scc_nongaussian_summary.csv', csv_nodes_path='scc_nongaussian_nodes.csv')

#NCC baseline mega study:
ncc_baseline_animation = sm.animate_displacement_vectors(sm.im_sim, sm.x_gt, sm.x_axis, sm.y_axis, sm.single_ncc, t_half_size, s_half_size, sm.mm_to_pixels, name='ncc', arrow_scale=20, fps=5)
ncc_baseline_errorplot = sm.plot_error_map_all_frames_csv(sm.im_sim, sm.x_gt, sm.x_axis, sm.y_axis, sm.single_ncc, t_half_size, s_half_size, sm.mm_to_pixels, save_path='ncc_error.png', csv_summary_path='ncc_summary.csv', csv_nodes_path='ncc_nodes.csv')

#not normalized scc mega study:
scc_not_normalized_animation = sm.animate_displacement_vectors(sm.im_sim, sm.x_gt, sm.x_axis, sm.y_axis, sm.single_scc_not_normalized, t_half_size, s_half_size, sm.mm_to_pixels, name='scc_not_normalized', arrow_scale=20, fps=5)
scc_not_normalized_errorplot = sm.plot_error_map_all_frames_csv(sm.im_sim, sm.x_gt, sm.x_axis, sm.y_axis, sm.single_scc_not_normalized, t_half_size, s_half_size, sm.mm_to_pixels, save_path='scc_not_normalized_error.png', csv_summary_path='scc_not_normalized_summary.csv', csv_nodes_path='scc_not_normalized_nodes.csv')