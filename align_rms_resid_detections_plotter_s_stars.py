#!/usr/bin/env python

# Starlist: magnitude and number comparison
# ---
# Abhimat Gautam

import numpy as np

from file_readers import stf_lis_reader,\
    align_orig_pos_reader, align_pos_reader, align_mag_reader

from astropy.table import Table
from astropy.io import fits

import os
from tqdm import tqdm
import warnings

dr_path = '/g/ghez/data/dr/dr1'
legacy_version_str = 'v2_3'
single_version_str = 'v3_1'
stf_corr = '0.8'

plate_scale = 0.00993   ## NIRC2 Plate Scale

def resid_detections_plotter(epoch_name, dr_path = '/g/ghez/data/dr/dr1',
                             filt_name = 'kp',
                             legacy_version_str = 'v2_3',
                             single_version_str = 'v3_1'):
    cur_wd = os.getcwd()
    
    # Analysis Data Location
    epoch_analysis_location = '{0}/{1}_{2}/'.format(cur_wd, epoch_name, filt_name)
    starlist_align_location = epoch_analysis_location + 'starlists_align_rms/align/'
    align_root = starlist_align_location + 'align_d_rms_abs'
    
    plot_out_dir = epoch_analysis_location + 'resid_detections_plots/'
    os.makedirs(plot_out_dir, exist_ok=True)
    
    stf_legacy_name = 'starfinder_' + legacy_version_str
    stf_singPSF_name = 'starfinder_' + single_version_str
    
    # Read in STF tables
    stf_legacy_lis_filename = '{0}/starlists/combo/{1}/{2}/mag{1}_{3}_rms.lis'.format(
        dr_path, epoch_name,
        stf_legacy_name,
        filt_name)

    stf_legacy_lis_table = stf_lis_reader(stf_legacy_lis_filename)

    stf_singPSF_lis_filename = '{0}/starlists/combo/{1}/{2}/mag{1}_{3}_rms.lis'.format(
        dr_path, epoch_name,
        stf_singPSF_name,
        filt_name)

    stf_singPSF_lis_table = stf_lis_reader(stf_singPSF_lis_filename)
    
    # Read in align tables
    align_orig_pos_table = align_orig_pos_reader(align_root + '.origpos',
                                                 align_stf_1_version=legacy_version_str,
                                                 align_stf_2_version=single_version_str)
    
    align_pos_table = align_orig_pos_reader(align_root + '.pos',
                                            align_stf_1_version=legacy_version_str,
                                            align_stf_2_version=single_version_str)
        
    # Determine closest star to Sgr A* and calculate center coordinates
    rad_distance = np.hypot(align_pos_table[single_version_str + '_x'],
                            align_pos_table[single_version_str + '_y'])
    
    close_star_index = np.argmin(rad_distance)
    close_star_op_x = (align_orig_pos_table[single_version_str + '_x'])[close_star_index]
    close_star_op_y = (align_orig_pos_table[single_version_str + '_y'])[close_star_index]
    close_star_pos_x = (align_pos_table[single_version_str + '_x'])[close_star_index]
    close_star_pos_y = (align_pos_table[single_version_str + '_y'])[close_star_index]
    
    center_op_x = close_star_op_x - (close_star_pos_x/plate_scale)
    center_op_y = close_star_op_y - (close_star_pos_y/plate_scale)
    
    # Common detections
    detection_filter = np.where(
        np.logical_and(align_orig_pos_table[legacy_version_str + '_x'] != -100000.0,
                       align_orig_pos_table[single_version_str + '_x'] != -100000.0))

    common_detections = align_orig_pos_table[detection_filter]
    common_detections_pos = align_pos_table[detection_filter]
    
    # Unique detections
    
    # Legacy detections only
    detection_filter = np.where(
        np.logical_and(align_orig_pos_table[legacy_version_str + '_x'] != -100000.0,
                       align_orig_pos_table[single_version_str + '_x'] == -100000.0))

    legacy_only_detections = align_orig_pos_table[detection_filter]
    legacy_only_detections_pos = align_pos_table[detection_filter]

    # Single PSF detections only
    detection_filter = np.where(
        np.logical_and(align_orig_pos_table[legacy_version_str + '_x'] == -100000.0,
                       align_orig_pos_table[single_version_str + '_x'] != -100000.0))

    singPSF_only_detections = align_orig_pos_table[detection_filter]
    singPSF_only_detections_pos = align_pos_table[detection_filter]
    
    # print(legacy_only_detections)
    # print(singPSF_only_detections)

    # Plot detections on the residual image
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as font_manager
    from matplotlib.ticker import MultipleLocator

    stf_res_versions = ['combo', 'legacy_res', 'singPSF_res', 'resDiff']
    
    # Load in the images
    
    fits_file = '{0}/combo/{1}/mag{1}_{2}.fits'.format(
        dr_path, epoch_name,
        filt_name)
    
    warnings.simplefilter('ignore', UserWarning)
    with fits.open(fits_file) as hdulist:
        combo_image_data = hdulist[0].data
    
    stf_version_name = stf_legacy_name
    fits_file = '{0}/starlists/combo/{1}/{2}/mag{1}_{3}_res.fits'.format(
        dr_path, epoch_name,
        stf_version_name, filt_name)
    
    warnings.simplefilter('ignore', UserWarning)
    with fits.open(fits_file) as hdulist:
        leg_res_image_data = hdulist[0].data
    
    stf_version_name = stf_singPSF_name
    fits_file = '{0}/starlists/combo/{1}/{2}/mag{1}_{3}_res.fits'.format(
        dr_path, epoch_name,
        stf_version_name, filt_name)
    
    warnings.simplefilter('ignore', UserWarning)
    with fits.open(fits_file) as hdulist:
        sin_res_image_data = hdulist[0].data
    
    
    for stf_res_version in stf_res_versions:
        fig, ax = plt.subplots(figsize=(5,5), frameon=False)
        
        ax.set_title(epoch_name.replace('_', '\_') + ' ' + filt_name)
        
        ax.axis('off')
        
        image_data = combo_image_data
        
        if stf_res_version == 'legacy_res':
            image_data = leg_res_image_data
        elif stf_res_version == 'singPSF_res':
            image_data = sin_res_image_data
        elif stf_res_version == 'resDiff':
            image_data = leg_res_image_data - sin_res_image_data
        
        # print(image_data[1100, 1100])
                
        # Default AO image scaling
        im_add = 0.
        im_floor = 100.
        im_ceil = 1.e6
        im_ceil = 1.e4
        im_ceil = 5.e3
        
        im_mult = 1.
        im_invert = 1.
        
        # Res File image scaling
        if stf_res_version != 'combo':
            im_add = 0.
            im_floor = -100.
            im_ceil = 400.

            im_floor = -1500.
            im_ceil = 1500.

            im_mult = 1.
            im_invert = 1.

        ## Put in image floor

        image_data[np.where(image_data <= im_floor)] = im_floor
        image_data[np.where(image_data >= im_ceil)] = im_ceil

        image_data = (image_data - im_floor)

        # print(np.min(image_data))
        # print(np.max(image_data))

        image_data *= im_mult

        # Display image
        im_cmap = plt.get_cmap('gray')
        
        if stf_res_version == 'resDiff':
            ax.imshow(im_invert * image_data,
                      cmap=im_cmap,
                      interpolation='nearest')
        else:
            ax.imshow(im_invert * np.sqrt(image_data),
                      cmap=im_cmap,
                      interpolation='nearest')
        ax.invert_yaxis()
        
        # Center on half arcsecond around Sgr A*
        arcsec_pixels = 0.5/plate_scale
        
        ax.set_xlim([center_op_x - arcsec_pixels,
                     center_op_x + arcsec_pixels])
        ax.set_ylim([center_op_y - arcsec_pixels,
                     center_op_y + arcsec_pixels])

        # Plot out detections
        circle_size=0.063 / 2
        
        x_label_offset = 4.5
        y_label_offset = 0.0
        
        label_rad = 0.3
        
        common_stars = []
        
        for star_index in range(len(common_detections)):
            x = common_detections[single_version_str + '_x'][star_index]
            y = common_detections[single_version_str + '_y'][star_index]
            
            r2d = np.hypot(
                common_detections_pos[single_version_str + '_x'][star_index],
                common_detections_pos[single_version_str + '_y'][star_index])
            
            c = ax.add_artist(plt.Circle((x,y),
                                         radius=(circle_size * 1./plate_scale),
                                         linestyle='-', edgecolor='greenyellow',   # , edgecolor='C0'
                                         label='Detections in both modes',
                                         linewidth=1.5, fill=False))
            common_stars.append(c)
            
            if r2d > label_rad:
                continue
            
            star_name = common_detections['name'][star_index]
            
            # if star_name not in name_stars:
            #     continue
            
            if star_name == 'star_894':
                star_name = 'Sgr A*'
            
            ax.text(
                x_label_offset + x,
                y_label_offset + y,
                star_name,
                ha='left', va='center', size='x-small',
                bbox = dict(boxstyle = 'round,pad=0.2', edgecolor='none',
                            facecolor = 'white', alpha = 0.5)
            )  # .replace('_', '\_')
            
        
        legacy_stars = []
        
        for star_index in range(len(legacy_only_detections)):
            x = legacy_only_detections[legacy_version_str + '_x'][star_index]
            y = legacy_only_detections[legacy_version_str + '_y'][star_index]
            
            r2d = np.hypot(
                legacy_only_detections_pos[legacy_version_str + '_x'][star_index],
                legacy_only_detections_pos[legacy_version_str + '_y'][star_index])
            
            c = ax.add_artist(plt.Circle((x, y),
                                         radius=(circle_size * 1./plate_scale),
                                         linestyle='-', edgecolor='C0',   # , edgecolor='C0'
                                         label='Legacy Only',
                                         linewidth=1.5, fill=False))
            legacy_stars.append(c)
            
            if r2d > label_rad:
                continue
            
            star_name = legacy_only_detections['name'][star_index]
            
            # if star_name not in name_stars:
            #     continue
            
            if star_name == 'star_894':
                star_name = 'Sgr A*'
            
            ax.text(
                x_label_offset + x,
                y_label_offset + y,
                star_name,
                ha='left', va='center', size='x-small',
                bbox = dict(boxstyle = 'round,pad=0.2', edgecolor='none',
                            facecolor = 'white', alpha = 0.5)
            )  # .replace('_', '\_')
        
        single_stars = []
        
        for star_index in range(len(singPSF_only_detections)):
            x = singPSF_only_detections[single_version_str + '_x'][star_index]
            y = singPSF_only_detections[single_version_str + '_y'][star_index]
            
            r2d = np.hypot(
                singPSF_only_detections_pos[single_version_str + '_x'][star_index],
                singPSF_only_detections_pos[single_version_str + '_y'][star_index])
            
            c = ax.add_artist(plt.Circle((x, y),
                                         radius=(circle_size * 1./plate_scale),
                                         linestyle='-', edgecolor='C1',
                                         label='Single-PSF Only',
                                         linewidth=1.5, fill=False))
            single_stars.append(c)
            
            if r2d > label_rad:
                continue
            
            star_name = singPSF_only_detections['name'][star_index]
            
            # if star_name not in name_stars:
            #     continue
            
            if star_name == 'star_894':
                star_name = 'Sgr A*'
            
            ax.text(
                x_label_offset + x,
                y_label_offset + y,
                star_name,
                ha='left', va='center', size='x-small',
                bbox = dict(boxstyle = 'round,pad=0.2', edgecolor='none',
                            facecolor = 'white', alpha = 0.5)
            )  # .replace('_', '\_')
        
        ax.legend(handles=[legacy_stars[0], single_stars[0], common_stars[0]],
                  loc='upper right', fontsize='small')
        
        # Save out and close figure
        fig.tight_layout()

        fig.savefig(plot_out_dir + 'resid_{0}_s_stars.pdf'.format(stf_res_version), transparent=True)
        fig.savefig(plot_out_dir + 'resid_{0}_s_stars.png'.format(stf_res_version), transparent=True, dpi=200)

        plt.close(fig)


# Read in epochs table
epochs_table = Table.read('epochs_table.h5', format='hdf5', path='data')

epochs_table = epochs_table[np.where(epochs_table['nights_combo'] == 'single_night')]

resid_detections_plotter('20160503nirc2', dr_path = dr_path,
                         filt_name = 'kp',
                         legacy_version_str = legacy_version_str,
                         single_version_str = single_version_str)


# # Run analysis code on all epochs
# for epochs_row in tqdm(epochs_table):
#     cur_epoch = epochs_row['epoch']
#     cur_filt = epochs_row['filt']
#
#     resid_detections_plotter(cur_epoch, dr_path = dr_path,
#                              filt_name = cur_filt,
#                              legacy_version_str = legacy_version_str,
#                              single_version_str = single_version_str)
#
#     # break
    