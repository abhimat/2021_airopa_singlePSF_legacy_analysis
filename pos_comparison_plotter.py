#!/usr/bin/env python

# Starlist magnitude and position comparison on the field
# ---
# Abhimat Gautam

import os

import numpy as np
from scipy import stats
from scipy.spatial import KDTree

from file_readers import stf_lis_reader,\
    align_orig_pos_reader, align_pos_reader, align_mag_reader

from astropy.table import Table

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

dr_path = '/g/ghez/data/dr/dr1'
legacy_version_str = 'v2_3'
single_version_str = 'v3_1'
stf_corr = '0.8'

plate_scale = 0.00993   ## NIRC2 Plate Scale

def analyze_pos_comparison(epoch_name, dr_path = '/g/ghez/data/dr/dr1',
                           filt_name = 'kp', stf_corr = '0.8',
                           legacy_version_str = 'v2_3',
                           single_version_str = 'v3_1',
                           mag_bin_lo = -1, mag_bin_hi = -1,
                           num_near_neighbors=20,
                          ):
    cur_wd = os.getcwd()
    
    epoch_analysis_location = '{0}/{1}_{2}/'.format(cur_wd, epoch_name, filt_name)
    starlist_align_location = epoch_analysis_location + 'starlists_align/align/'
    align_root = starlist_align_location + 'align_d_abs'
    
    plot_out_dir = epoch_analysis_location + 'pos_comparison_plots/'
    os.makedirs(plot_out_dir, exist_ok=True)
    
    out_mag_suffix = ''
    if mag_bin_lo != -1:
        out_mag_suffix = f'_mag_{mag_bin_lo}_{mag_bin_hi}'
    
    stf_legacy_name = 'starfinder_' + legacy_version_str
    stf_singPSF_name = 'starfinder_' + single_version_str
    
    # Read in STF tables
    stf_legacy_lis_filename = '{0}/starlists/combo/{1}/{2}/mag{1}_{3}_{4}_stf.lis'.format(
        dr_path, epoch_name,
        stf_legacy_name,
        filt_name, stf_corr)

    stf_legacy_lis_table = stf_lis_reader(stf_legacy_lis_filename)

    stf_singPSF_lis_filename = '{0}/starlists/combo/{1}/{2}/mag{1}_{3}_{4}_stf.lis'.format(
        dr_path, epoch_name,
        stf_singPSF_name,
        filt_name, stf_corr)

    stf_singPSF_lis_table = stf_lis_reader(stf_singPSF_lis_filename)
    
    # Read in align tables
    align_mag_table = align_mag_reader(align_root + '.mag',
                                       align_stf_1_version=legacy_version_str,
                                       align_stf_2_version=single_version_str)

    align_pos_table = align_pos_reader(align_root + '.pos',
                                       align_stf_1_version=legacy_version_str,
                                       align_stf_2_version=single_version_str)

    align_orig_pos_table = align_orig_pos_reader(align_root + '.origpos',
                                                 align_stf_1_version=legacy_version_str,
                                                 align_stf_2_version=single_version_str)
    
    # Common detections in each starlist
    
    
    det_both = np.logical_and(align_mag_table[legacy_version_str + '_mag'] != 0.0,
                       align_mag_table[single_version_str + '_mag'] != 0.0)
    
    detection_filter = np.where(det_both)
    # print(detection_filter)
    
    if mag_bin_lo != -1:
        det_mag = np.logical_and(align_mag_table[single_version_str + '_mag'] >= mag_bin_lo,
                       align_mag_table[single_version_str + '_mag'] < mag_bin_hi)
        
        detection_filter = np.where(np.logical_and(det_both, det_mag))
        
        # print(detection_filter)
    
    common_detections_mag_table = align_mag_table[detection_filter]
    common_detections_pos_table = align_pos_table[detection_filter]
    common_detections_orig_pos_table = align_orig_pos_table[detection_filter]
    
    # Bright stars for reference
    if mag_bin_lo == -1:
        bright_cutoff = 12.
    
        bright_filter = np.where(common_detections_mag_table[single_version_str + '_mag'] <= bright_cutoff)
    
        bright_pos_table = common_detections_pos_table[bright_filter]
        bright_orig_pos_table = common_detections_orig_pos_table[bright_filter]
    
    # Pos differences
    diff_pos_x = (common_detections_orig_pos_table[single_version_str + '_x'] -
                  common_detections_orig_pos_table[legacy_version_str + '_x'])
    diff_pos_y = (common_detections_orig_pos_table[single_version_str + '_y'] -
                  common_detections_orig_pos_table[legacy_version_str + '_y'])

    common_x = common_detections_orig_pos_table[legacy_version_str + '_x']
    common_y = common_detections_orig_pos_table[legacy_version_str + '_y']
    
    if len(diff_pos_x) < 1:
        return
    
    # Position comparison
    fig, (ax1, ax2) = plt.subplots(figsize=(8,4), nrows=1, ncols=2)
    
    ax1.set_title(epoch_name.replace('_', '\_') + ' ' + filt_name)
    
    ax1.plot(common_detections_orig_pos_table[legacy_version_str + '_x'],
             common_detections_orig_pos_table[single_version_str + '_x'],
             '.', color='k', alpha=0.6)
    
    ax2.plot(common_detections_orig_pos_table[legacy_version_str + '_y'],
             common_detections_orig_pos_table[single_version_str + '_y'],
             '.', color='k', alpha=0.6)
    
    ax1.set_xlabel(r"Legacy Starfinder: $x$")
    ax1.set_ylabel(r"Single PSF Starfinder: $x$")
    
    ax2.set_xlabel(r"Legacy Starfinder: $y$")
    ax2.set_ylabel(r"Single PSF Starfinder: $y$")
    
    ax1.set_aspect('equal', 'box')
    ax2.set_aspect('equal', 'box')
    
    fig.tight_layout()

    fig.savefig(f"{plot_out_dir}stf_pos_comparison{out_mag_suffix}.pdf")
    fig.savefig(f"{plot_out_dir}stf_pos_comparison{out_mag_suffix}.png", dpi=200)

    plt.close(fig)
    
    # Delta position comparison
    fig, (ax1, ax2) = plt.subplots(figsize=(8,4), nrows=1, ncols=2)
    
    ax1.set_title(epoch_name.replace('_', '\_') + ' ' + filt_name)
    
    ax1.plot(common_detections_orig_pos_table[legacy_version_str + '_x'],
             diff_pos_x,
             '.', color='k', alpha=0.6)
    
    ax2.plot(common_detections_orig_pos_table[legacy_version_str + '_y'],
             diff_pos_y,
             '.', color='k', alpha=0.6)
    
    ax1.set_xlabel(r"Legacy Starfinder: $x$")
    ax1.set_ylabel(r"Single PSF $x$ $-$ Legacy $x$")
    
    ax2.set_xlabel(r"Legacy Starfinder: $y$")
    ax2.set_ylabel(r"Single PSF $y$ $-$ Legacy $y$")
    
    ax1.set_ylim([np.median(diff_pos_x) - 5.*stats.median_abs_deviation(diff_pos_x),
                  np.median(diff_pos_x) + 5.*stats.median_abs_deviation(diff_pos_x)])
    
    ax1.axhline(np.median(diff_pos_x), color='k', ls='-')
    ax1.axhline(np.median(diff_pos_x) + stats.median_abs_deviation(diff_pos_x), color='k', ls='--')
    ax1.axhline(np.median(diff_pos_x) - stats.median_abs_deviation(diff_pos_x), color='k', ls='--')
    
    ax2.set_ylim([np.median(diff_pos_y) - 5.*stats.median_abs_deviation(diff_pos_y),
                  np.median(diff_pos_y) + 5.*stats.median_abs_deviation(diff_pos_y)])
    
    ax2.axhline(np.median(diff_pos_y), color='k', ls='-')
    ax2.axhline(np.median(diff_pos_y) + stats.median_abs_deviation(diff_pos_y), color='k', ls='--')
    ax2.axhline(np.median(diff_pos_y) - stats.median_abs_deviation(diff_pos_y), color='k', ls='--')
    
    
    fig.tight_layout()

    fig.savefig(f"{plot_out_dir}stf_pos_delta_comparison{out_mag_suffix}.pdf")
    fig.savefig(f"{plot_out_dir}stf_pos_delta_comparison{out_mag_suffix}.png", dpi=200)

    plt.close(fig)
    
    # Quiver plot
    fig, ax = plt.subplots(figsize=(4.8,5.2))
    
    ax.set_title(epoch_name.replace('_', '\_') + ' ' + filt_name)
    
    q = ax.quiver(common_detections_pos_table[legacy_version_str + '_x'] * -1,
                  common_detections_pos_table[legacy_version_str + '_y'],
                  diff_pos_x, diff_pos_y,
                  scale=0.25, scale_units='xy')
    
    ax.quiverkey(q, X=0.05, Y=-0.15, U=0.1, labelpos='E',
                 label=r'0.1 pixel, Single PSF $-$ Legacy Starfinder position',
                 fontproperties={'size':'x-small'})
    
    if mag_bin_lo == -1:
        ax.plot(bright_pos_table[legacy_version_str + '_x'] * -1,
                bright_pos_table[legacy_version_str + '_y'],
                'o', color='royalblue', ms=2.0)
            
    ax.set_aspect('equal', 'box')
    
    ax.set_xlabel(r"arcsec E of Sgr A*")
    ax.set_ylabel(r"arcsec N of Sgr A*")
    
    ax.set_xlim([6, -6])
    ax.set_ylim([-7, 5])
    
    x_majorLocator = MultipleLocator(2)
    x_minorLocator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)
    
    y_majorLocator = MultipleLocator(2)
    y_minorLocator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    # Save out and close figure
    fig.tight_layout()

    fig.savefig(f'{plot_out_dir}stf_pos_quiv_comparison{out_mag_suffix}.pdf')
    fig.savefig(f'{plot_out_dir}stf_pos_quiv_comparison{out_mag_suffix}.png', dpi=200)
    
    plt.close(fig)



    # Nearest neighbors plot
    ## Construct kd-tree for quick lookup for closest neighbors
    neighbor_kdtree = KDTree(list(zip(common_x, common_y)))

    ## Construct grid points for nearest neighbors map
    near_neighbors_map_x_bounds = [0, 1200]
    near_neighbors_map_y_bounds = [0, 1200]

    near_neighbors_map_grid_spacing = 20

    near_neighbors_map_x_coords = np.arange(near_neighbors_map_x_bounds[0], near_neighbors_map_x_bounds[1] + near_neighbors_map_grid_spacing, near_neighbors_map_grid_spacing)
    near_neighbors_map_y_coords = np.arange(near_neighbors_map_y_bounds[0], near_neighbors_map_y_bounds[1] + near_neighbors_map_grid_spacing, near_neighbors_map_grid_spacing)

    near_neighbors_map_plot_x, near_neighbors_map_plot_y = np.meshgrid(near_neighbors_map_x_coords, near_neighbors_map_y_coords)

    ## Go through each grid point
    ### Function to evaluate mean and median value of nearest neighbors
    def near_neighbors_mean_median_val(x_coord, y_coord):
        #### Find nearest neighbor stars
        near_neighbors = neighbor_kdtree.query([x_coord, y_coord], k=num_near_neighbors)
        near_neighbors_coords, near_neighbors_indices = near_neighbors

        #### Compute mean reduced chi squared of the nearest neighbors
        val_x_array = np.empty(len(near_neighbors_indices))
        val_y_array = np.empty(len(near_neighbors_indices))

        for index in range(len(near_neighbors_indices)):
            cur_neighbor_index = near_neighbors_indices[index]
        
            val_x_array[index] = diff_pos_x[cur_neighbor_index]
            val_y_array[index] = diff_pos_y[cur_neighbor_index]

        mean_val_x = np.mean(val_x_array)
        median_val_x = np.median(val_x_array)
    
        mean_val_y = np.mean(val_y_array)
        median_val_y = np.median(val_y_array)

        return (mean_val_x, median_val_x,
                mean_val_y, median_val_y)

    ### Vectorized version of near_neighbors_mean_median_rcs function
    vector_near_neighbors_mean_median_val = np.vectorize(near_neighbors_mean_median_val)

    ### Evaluate over all grid points
    (near_neighbors_map_plot_mean_val_x,
     near_neighbors_map_plot_median_val_x,
     near_neighbors_map_plot_mean_val_y,
     near_neighbors_map_plot_median_val_y) = vector_near_neighbors_mean_median_val(
                                               near_neighbors_map_plot_x,
                                               near_neighbors_map_plot_y)
    
    # Compute a transformation from pixels to abs coordinates
    # using nearest star to Sgr A*
    rad_distance = np.hypot(common_detections_pos_table[legacy_version_str + '_x'],
                            common_detections_pos_table[legacy_version_str + '_y'])
    
    close_star_index = np.argmin(rad_distance)
    close_star_op_x = (common_detections_orig_pos_table[legacy_version_str + '_x'])[close_star_index]
    close_star_op_y = (common_detections_orig_pos_table[legacy_version_str + '_y'])[close_star_index]
    close_star_pos_x = (common_detections_pos_table[legacy_version_str + '_x'])[close_star_index]
    close_star_pos_y = (common_detections_pos_table[legacy_version_str + '_y'])[close_star_index]
    
    center_op_x = close_star_op_x - (close_star_pos_x/plate_scale)
    center_op_y = close_star_op_y - (close_star_pos_y/plate_scale)
    
    
    # Quiver plot
    fig, ax = plt.subplots(figsize=(4.8,5.2))
    
    ax.set_title(epoch_name.replace('_', '\_') + ' ' + filt_name)
    
    q = ax.quiver((near_neighbors_map_plot_x - center_op_x) * plate_scale * -1,
                  (near_neighbors_map_plot_y - center_op_y) * plate_scale,
                  near_neighbors_map_plot_median_val_x,
                  near_neighbors_map_plot_median_val_y,
                  scale=0.25, scale_units='xy')

    quiver_label = r'0.1 pixel, Single PSF $-$ Legacy Starfinder position'
    quiver_label += ' (median of 20 nearest neighbors)'

    ax.quiverkey(q, X=0.05, Y=-0.15, U=0.1, labelpos='E',
                 label=quiver_label,
                 fontproperties={'size':'x-small'})
    
    if mag_bin_lo == -1:
        ax.plot(bright_pos_table[legacy_version_str + '_x'] * -1,
                bright_pos_table[legacy_version_str + '_y'],
                'o', color='royalblue', ms=2.5)

    ax.set_aspect('equal', 'box')

    ax.set_xlabel(r"arcsec E of Sgr A*")
    ax.set_ylabel(r"arcsec N of Sgr A*")
    
    ax.set_xlim([6, -6])
    ax.set_ylim([-7, 5])

    x_majorLocator = MultipleLocator(2)
    x_minorLocator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(2)
    y_minorLocator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)


    # Save out and close figure
    fig.tight_layout()
    
    out_mag_suffix = ''
    if mag_bin_lo != -1:
        out_mag_suffix = f'_mag_{mag_bin_lo}_{mag_bin_hi}'

    fig.savefig(f'{plot_out_dir}stf_pos_quiv_comparison_nearneigh{out_mag_suffix}.pdf')
    fig.savefig(f'{plot_out_dir}stf_pos_quiv_comparison_nearneigh{out_mag_suffix}.png', dpi=200)

    plt.close(fig)


# Read in epochs table
epochs_table = Table.read('epochs_table.h5', format='hdf5', path='data')

epochs_table = epochs_table[np.where(epochs_table['nights_combo'] == 'single_night')]

# Run analysis code on all epochs
for epochs_row in tqdm(epochs_table[:30]):
    cur_epoch = epochs_row['epoch']
    print(cur_epoch)
    cur_filt = epochs_row['filt']
    
    analyze_pos_comparison(cur_epoch, dr_path = dr_path,
                           filt_name = cur_filt, stf_corr = stf_corr,
                           legacy_version_str = legacy_version_str,
                           single_version_str = single_version_str)
    
    
    # Run this comparison in different mag bins
    mag_bins_lo = [11, 13, 15, 17]
    mag_bins_hi = [13, 15, 17, 25]
    
    num_near_neighbors_vals = [5, 10, 20, 10]
    
    if cur_filt == 'h':
        mag_bins_lo = [13, 15, 17, 19]
        mag_bins_hi = [15, 17, 19, 27]
    
    for (cur_bin_lo, cur_bin_hi,
         num_near_neighbors,
        ) in zip(mag_bins_lo, mag_bins_hi,
                 num_near_neighbors_vals):
        analyze_pos_comparison(cur_epoch, dr_path = dr_path,
                               filt_name = cur_filt, stf_corr = stf_corr,
                               legacy_version_str = legacy_version_str,
                               single_version_str = single_version_str,
                               mag_bin_lo = cur_bin_lo, mag_bin_hi = cur_bin_hi,
                               num_near_neighbors = num_near_neighbors,
                              )
    
    # break
