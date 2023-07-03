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
filters = ['kp',]

skip_epochs = []

filt_mag_bins = {}

filt_mag_bins['kp'] = [
    '11_13',
    '13_15',
    '15_17',
    '17_25',
]

filt_mag_bins['h'] = [
    '13_13',
    '15_15',
    '17_17',
    '19_27',
]

for filter in filters:
    # Directories
    cur_wd = os.getcwd()

    plot_out_dir = cur_wd + '/pos_comparison_plots/'
    os.makedirs(plot_out_dir, exist_ok=True)
    
    # Read in epochs table
    epochs_table = Table.read('epochs_table.h5', format='hdf5', path='data')
    
    epochs_table = epochs_table[np.where(epochs_table['nights_combo'] == 'single_night')]
    epochs_table = epochs_table[np.where(epochs_table['filt'] == filter)]
    
    # Set up GIF writers
    comp_gif_writer = imageio.get_writer(
        '{0}/stf_pos_comparison_{1}.gif'.format(plot_out_dir, filter),
        mode='I', fps=1, subrectangles=True,
    )
    comp_nearneigh_gif_writer = imageio.get_writer(
        '{0}/stf_pos_comparison_nearneigh_{1}.gif'.format(plot_out_dir, filter),
        mode='I', fps=1, subrectangles=True,
    )
    
    mag_bins = filt_mag_bins[filter]
    
    bin_writers = []
    
    for mag_bin in mag_bins:
        cur_writer = imageio.get_writer(
            '{0}/stf_pos_comparison_nearneigh_mag_{2}_{1}.gif'.format(
                plot_out_dir, filter, mag_bin
            ),
            mode='I', fps=1, subrectangles=True,
        )
        
        bin_writers.append(cur_writer)
    
    
    
    # Pull image from all epochs and append to GIF
    for epochs_row in tqdm(epochs_table):
        cur_epoch = epochs_row['epoch']
        cur_filt = epochs_row['filt']
    
        print(cur_epoch)
        if cur_epoch in skip_epochs:
            continue
        
        cur_epoch_analysis_location = '{0}/{1}_{2}/'.format(cur_wd, cur_epoch,
                                                            cur_filt)
        cur_plot_dir = cur_epoch_analysis_location + 'pos_comparison_plots/'
    
        cur_plot_file = cur_plot_dir + 'stf_pos_comparison.png'
        cur_image = imageio.imread(cur_plot_file)
        comp_gif_writer.append_data(cur_image)
    
        cur_plot_file = cur_plot_dir + 'stf_pos_comparison_nearneigh.png'
        cur_image = imageio.imread(cur_plot_file)
        comp_nearneigh_gif_writer.append_data(cur_image)
        
        for (mag_bin, bin_writer) in zip(mag_bins, bin_writers):
            cur_plot_file = cur_plot_dir + 'stf_pos_comparison_nearneigh_mag_{0}.png'.format(
                mag_bin
            )
            cur_image = imageio.imread(cur_plot_file)
            bin_writer.append_data(cur_image)

    # Close out GIF writers
    comp_gif_writer.close()
    comp_nearneigh_gif_writer.close()
    
    for bin_writer in bin_writers:
        bin_writer.close()
