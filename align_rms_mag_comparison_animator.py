#!/usr/bin/env python

# Starlist: magnitude and number comparison
# ---
# Abhimat Gautam

import numpy as np

from file_readers import stf_lis_reader,\
    align_orig_pos_reader, align_pos_reader, align_mag_reader

from astropy.table import Table
from astropy.io import fits

import imageio

import os
from tqdm import tqdm
import warnings

dr_path = '/g/ghez/data/dr/dr1'
legacy_version_str = 'v2_3'
single_version_str = 'v3_1'
stf_corr = '0.8'

stf_res_versions = ['legacy', 'singPSF']

plate_scale = 0.00993   ## NIRC2 Plate Scale

filters = ['kp', 'h']

for filter in filters:
    # Directories
    cur_wd = os.getcwd()

    plot_out_dir = cur_wd + '/align_rms_mag_comparison_plots/'
    os.makedirs(plot_out_dir, exist_ok=True)
    
    # Read in epochs table
    epochs_table = Table.read('epochs_table.h5', format='hdf5', path='data')
    
    epochs_table = epochs_table[np.where(epochs_table['nights_combo'] == 'single_night')]
    epochs_table = epochs_table[np.where(epochs_table['filt'] == filter)]
    
    # Set up GIF writers
    stf_lum_writer = imageio.get_writer(
                            '{0}/stf_lum_funcs_{1}.gif'.format(plot_out_dir, filter),
                            mode='I', fps=1, subrectangles=True)
    
    stf_lum_one_mode_writer = imageio.get_writer(
                            '{0}/stf_lum_funcs_one_mode_{1}.gif'.format(plot_out_dir, filter),
                            mode='I', fps=1, subrectangles=True)
    
    comp_gif_writer = imageio.get_writer(
                            '{0}/stf_mag_comparison_{1}.gif'.format(plot_out_dir, filter),
                            mode='I', fps=1, subrectangles=True)
    
    mag_delta_gif_writer = imageio.get_writer(
                            '{0}/stf_mag_delta_comparison_{1}.gif'.format(plot_out_dir, filter),
                            mode='I', fps=1, subrectangles=True)
    
    mag_delta_nostars_gif_writer = imageio.get_writer(
                            '{0}/stf_mag_delta_comparison_nostars_{1}.gif'.format(plot_out_dir, filter),
                            mode='I', fps=1, subrectangles=True)
    
    # comp_nearneigh_gif_writer = imageio.get_writer(
    #                         '{0}/stf_pos_comparison_nearneigh_{1}.gif'.format(plot_out_dir, filter),
    #                         mode='I', fps=1, subrectangles=True)

    # Pull image from all epochs and append to GIF
    for epochs_row in tqdm(epochs_table[2:]):
        cur_epoch = epochs_row['epoch']
        cur_filt = epochs_row['filt']
    
        cur_epoch_analysis_location = '{0}/{1}_{2}/'.format(cur_wd, cur_epoch,
                                                            cur_filt)
        cur_plot_dir = cur_epoch_analysis_location + 'align_rms_mag_comparison_plots/'
    
        cur_plot_file = cur_plot_dir + 'stf_lum_funcs.png'
        cur_image = imageio.imread(cur_plot_file)
        stf_lum_writer.append_data(cur_image)
        
        cur_plot_file = cur_plot_dir + 'stf_lum_funcs_one_mode.png'
        cur_image = imageio.imread(cur_plot_file)
        stf_lum_one_mode_writer.append_data(cur_image)
        
        cur_plot_file = cur_plot_dir + 'stf_mag_comparison.png'
        cur_image = imageio.imread(cur_plot_file)
        comp_gif_writer.append_data(cur_image)
        
        cur_plot_file = cur_plot_dir + 'stf_mag_delta_comparison.png'
        cur_image = imageio.imread(cur_plot_file)
        mag_delta_gif_writer.append_data(cur_image)
        
        cur_plot_file = cur_plot_dir + 'stf_mag_delta_comparison_nostars.png'
        cur_image = imageio.imread(cur_plot_file)
        mag_delta_nostars_gif_writer.append_data(cur_image)
        
        # cur_plot_file = cur_plot_dir + 'stf_pos_comparison_nearneigh.png'
        # cur_image = imageio.imread(cur_plot_file)
        # comp_nearneigh_gif_writer.append_data(cur_image)

    # Close out GIF writers
    stf_lum_writer.close()
    stf_lum_one_mode_writer.close()
    comp_gif_writer.close()
    mag_delta_gif_writer.close()
    mag_delta_nostars_gif_writer.close()
    