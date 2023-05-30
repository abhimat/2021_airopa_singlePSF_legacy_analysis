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
filters = ['kp']

for filter in filters:
    # Directories
    cur_wd = os.getcwd()

    plot_out_dir = cur_wd + '/align_rms_pos_num_comparison_plots/'
    os.makedirs(plot_out_dir, exist_ok=True)
    
    # Read in epochs table
    epochs_table = Table.read('epochs_table.h5', format='hdf5', path='data')
    
    epochs_table = epochs_table[np.where(epochs_table['nights_combo'] == 'single_night')]
    epochs_table = epochs_table[np.where(epochs_table['filt'] == filter)]
    
    # Set up GIF writers
    magnumhist_gif_writer = imageio.get_writer(
        '{0}/mag_num_hist_{1}.gif'.format(plot_out_dir, filter),
        mode='I', fps=1, subrectangles=True)
    magnumhist_ca_gif_writer = imageio.get_writer(
        '{0}/mag_num_hist_centarcsec_{1}.gif'.format(plot_out_dir, filter),
        mode='I', fps=1, subrectangles=True)
    # comp_nearneigh_gif_writer = imageio.get_writer(
    #                         '{0}/stf_pos_comparison_nearneigh_{1}.gif'.format(plot_out_dir, filter),
    #                         mode='I', fps=1, subrectangles=True)
    
    # Pull image from all epochs and append to GIF
    epochs_list = list(range(0,78)) + list(range(79, 92))
    
    for epochs_row in tqdm(epochs_table[epochs_list]):
        cur_epoch = epochs_row['epoch']
        cur_filt = epochs_row['filt']
        
        cur_epoch_analysis_location = '{0}/{1}_{2}/'.format(cur_wd, cur_epoch,
                                                            cur_filt)
        cur_plot_dir = cur_epoch_analysis_location + 'align_rms_pos_num_comparison_plots/'
        
        cur_plot_file = cur_plot_dir + 'mag_num_hist.png'
        cur_image = imageio.imread(cur_plot_file)
        magnumhist_gif_writer.append_data(cur_image)
        
        cur_plot_file = cur_plot_dir + 'mag_num_hist_centarcsec.png'
        cur_image = imageio.imread(cur_plot_file)
        magnumhist_ca_gif_writer.append_data(cur_image)
        #
        # cur_plot_file = cur_plot_dir + 'stf_pos_comparison_nearneigh.png'
        # cur_image = imageio.imread(cur_plot_file)
        # comp_nearneigh_gif_writer.append_data(cur_image)

    # Close out GIF writers
    magnumhist_gif_writer.close()
    magnumhist_ca_gif_writer.close()
