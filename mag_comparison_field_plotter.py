#!/usr/bin/env python

# Starlist: magnitude comparison on the field
# ---
# Abhimat Gautam

import os

import numpy as np
from scipy import stats
from scipy.spatial import KDTree

from file_readers import stf_lis_reader,\
    align_orig_pos_reader, align_pos_reader, align_mag_reader

from astropy.table import Table
from astropy.io import fits

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import warnings

dr_path = '/g/ghez/data/dr/dr1'
legacy_version_str = 'v2_3'
single_version_str = 'v3_1'
stf_corr = '0.8'

def analyze_mag_field_comparison(epoch_name, dr_path = '/g/ghez/data/dr/dr1',
                                 filt_name = 'kp', stf_corr = '0.8',
                                 legacy_version_str = 'v2_3',
                                 single_version_str = 'v3_1'):
    cur_wd = os.getcwd()
    
    epoch_analysis_location = '{0}/{1}_{2}/'.format(cur_wd, epoch_name, filt_name)
    starlist_align_location = epoch_analysis_location + 'starlists_align/align/'
    align_root = starlist_align_location + 'align_d'
    
    plot_out_dir = epoch_analysis_location + 'mag_comparison_plots/'
    os.makedirs(plot_out_dir, exist_ok=True)
    
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
    detection_filter = np.where(
        np.logical_and(align_mag_table[legacy_version_str + '_mag'] != 0.0,
                       align_mag_table[single_version_str + '_mag'] != 0.0))
    
    common_detections_mag_table = align_mag_table[detection_filter]
    common_detections_pos_table = align_pos_table[detection_filter]
    common_detections_orig_pos_table = align_orig_pos_table[detection_filter]

    # Mag differences and binned mag differences
    diff_mag = (common_detections_mag_table[single_version_str + '_mag'] -
                common_detections_mag_table[legacy_version_str + '_mag'])

    common_x = common_detections_pos_table[single_version_str + '_x']
    common_y = common_detections_pos_table[single_version_str + '_y']
    
    
    # Nearest neighbors mag difference plot
    num_near_neighbors = 20
    
    ## Construct kd-tree for quick lookup for closest neighbors
    neighbor_kdtree = KDTree(list(zip(common_x, common_y)))

    ## Construct grid points for nearest neighbors map
    near_neighbors_map_x_bounds = [0, 1200]
    near_neighbors_map_y_bounds = [0, 1200]

    near_neighbors_map_grid_spacing = 10

    near_neighbors_map_x_coords = np.arange(near_neighbors_map_x_bounds[0], near_neighbors_map_x_bounds[1] + near_neighbors_map_grid_spacing, near_neighbors_map_grid_spacing)
    near_neighbors_map_y_coords = np.arange(near_neighbors_map_y_bounds[0], near_neighbors_map_y_bounds[1] + near_neighbors_map_grid_spacing, near_neighbors_map_grid_spacing)

    near_neighbors_map_plot_x, near_neighbors_map_plot_y = np.meshgrid(near_neighbors_map_x_coords, near_neighbors_map_y_coords)

    ## Go through each grid point
    ### Function to evaluate mean and median reduced chi squared of nearest neighbors
    def near_neighbors_mean_median_val(x_coord, y_coord):
        #### Find nearest neighbor stars
        near_neighbors = neighbor_kdtree.query([x_coord, y_coord], k=num_near_neighbors)
        near_neighbors_coords, near_neighbors_indices = near_neighbors
    
        #### Compute mean reduced chi squared of the nearest neighbors
        val_array = np.empty(len(near_neighbors_indices))
    
        for index in range(len(near_neighbors_indices)):
            cur_neighbor_index = near_neighbors_indices[index]
        
            val_array[index] = diff_mag[cur_neighbor_index]
    
        mean_val = np.mean(val_array)
        median_val = np.median(val_array)
    
        return mean_val, median_val

    ### Vectorized version of near_neighbors_mean_median_rcs function
    vector_near_neighbors_mean_median_val = np.vectorize(near_neighbors_mean_median_val)

    ### Evaluate over all grid points
    (near_neighbors_map_plot_mean_val,
     near_neighbors_map_plot_median_val) = vector_near_neighbors_mean_median_val(
                                               near_neighbors_map_plot_x,
                                               near_neighbors_map_plot_y)

    ## Draw nearest neighbors maps
    ### Flip x coordinates to match E direction
    near_neighbors_map_x_bounds = [0, 1200]
    near_neighbors_map_y_bounds = [0, 1200]

    map_extents = [near_neighbors_map_x_bounds[0] + near_neighbors_map_grid_spacing,
                   near_neighbors_map_x_bounds[1] - near_neighbors_map_grid_spacing,
                   near_neighbors_map_y_bounds[0] - near_neighbors_map_grid_spacing,
                   near_neighbors_map_y_bounds[1] + near_neighbors_map_grid_spacing]


    fig, ax = plt.subplots(figsize=(4.5,5))

    color_normalizer = mpl.colors.Normalize(vmin=-0.1, vmax=0.1)
    color_cmap = plt.get_cmap('plasma')

    map1_fill = ax.imshow(near_neighbors_map_plot_median_val,
                           cmap=color_cmap, norm=color_normalizer,
                           extent=map_extents)

    ### Flip x coordinates to match E direction
    near_neighbors_map_x_bounds = [0, 1200]

    ax.axis('equal')

    ax.set_xlabel(r"Single PSF Starfinder $x$ (pixels)")
    ax.set_ylabel(r"Single PSF Starfinder $x$ (pixels)")

    ax.set_xlim(near_neighbors_map_x_bounds)
    ax.set_ylim(near_neighbors_map_y_bounds)


    x_majorLocator = MultipleLocator(200)
    x_minorLocator = MultipleLocator(50)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(200)
    y_minorLocator = MultipleLocator(50)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)

    # Color bar
    sc_map = mpl.cm.ScalarMappable(norm=color_normalizer, cmap=color_cmap)

    plt.colorbar(sc_map, ax=ax, orientation='horizontal',
                 label=r"Single PSF $m_{K'}$ $-$ Legacy $m_{K'}$")

    # Save out and close figure
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'stf_mag_delta_field_comparison.pdf')
    fig.savefig(plot_out_dir + 'stf_mag_delta_field_comparison.png', dpi=200)

    plt.close(fig)

# Read in epochs table
epochs_table = Table.read('epochs_table.h5', format='hdf5', path='data')

epochs_table = epochs_table[np.where(epochs_table['nights_combo'] == 'single_night')]

# Run analysis code on all epochs
for epochs_row in tqdm(epochs_table):
    cur_epoch = epochs_row['epoch']
    cur_filt = epochs_row['filt']
    
    analyze_mag_field_comparison(cur_epoch, dr_path = dr_path,
                                 filt_name = cur_filt, stf_corr = stf_corr,
                                 legacy_version_str = legacy_version_str,
                                 single_version_str = single_version_str)
    
