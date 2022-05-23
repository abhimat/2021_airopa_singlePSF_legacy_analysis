#!/usr/bin/env python

# Starlist: magnitude comparison
# ---
# Abhimat Gautam

import os

import numpy as np
from scipy import stats

from file_readers import stf_rms_lis_reader,\
    align_orig_pos_reader, align_pos_reader, align_pos_err_reader,\
    align_mag_reader

from astropy.table import Table
from astropy.io import fits

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import warnings

dr_path = '/g/ghez/data/dr/dr1'
legacy_version_str = 'v2_3'
single_version_str = 'v3_1'
stf_corr = '0.8'

filt_label_strs = {'kp': r"$K'$-band",
                   'h': r"$H$-band",
                   'lp': r"$L'$-band",
                  }

filt_mag_label_strs = {'kp': r"$m_{K'}$",
                       'h': r"$m_{H}$",
                       'lp': r"$m_{L'}$",
                      }

def analyze_mag_comparison(epoch_name, dr_path = '/g/ghez/data/dr/dr1',
                           filt_name = 'kp',
                           legacy_version_str = 'v2_3',
                           single_version_str = 'v3_1'):
    cur_wd = os.getcwd()
    
    epoch_analysis_location = '{0}/{1}_{2}/'.format(cur_wd, epoch_name, filt_name)
    starlist_align_location = epoch_analysis_location + 'starlists_align_rms/align/'
    align_root = starlist_align_location + 'align_d_rms_abs'
    
    plot_out_dir = epoch_analysis_location + 'align_rms_mag_comparison_plots/'
    os.makedirs(plot_out_dir, exist_ok=True)
    
    stf_legacy_name = 'starfinder_' + legacy_version_str
    stf_singPSF_name = 'starfinder_' + single_version_str
    
    # Read in STF tables
    stf_legacy_lis_filename = '{0}/starlists/combo/{1}/{2}/mag{1}_{3}_rms.lis'.format(
        dr_path, epoch_name,
        stf_legacy_name,
        filt_name)

    stf_legacy_lis_table = stf_rms_lis_reader(stf_legacy_lis_filename)

    stf_singPSF_lis_filename = '{0}/starlists/combo/{1}/{2}/mag{1}_{3}_rms.lis'.format(
        dr_path, epoch_name,
        stf_singPSF_name,
        filt_name)

    stf_singPSF_lis_table = stf_rms_lis_reader(stf_singPSF_lis_filename)
    
    
    # Read in align tables
    align_mag_table = align_mag_reader(align_root + '.mag',
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
    common_detections_orig_pos_table = align_orig_pos_table[detection_filter]
    
    # One mode detection
    leg_detection_filter = np.where(
        np.logical_and(align_mag_table[legacy_version_str + '_mag'] != 0.0,
                       align_mag_table[single_version_str + '_mag'] == 0.0))
    
    sing_detection_filter = np.where(
        np.logical_and(align_mag_table[legacy_version_str + '_mag'] == 0.0,
                       align_mag_table[single_version_str + '_mag'] != 0.0))
    
    legonly_detections_mag_table = align_mag_table[leg_detection_filter]
    legonly_detections_orig_pos_table = align_orig_pos_table[leg_detection_filter]
    
    singonly_detections_mag_table = align_mag_table[sing_detection_filter]
    singonly_detections_orig_pos_table = align_orig_pos_table[sing_detection_filter]
    
    # Mag differences and binned mag differences
    diff_mag = (common_detections_mag_table[single_version_str + '_mag'] -
                common_detections_mag_table[legacy_version_str + '_mag'])

    diff_mag_median_bin_cents = np.arange(9.5, 21., 0.5)
    diff_mag_median_bin_size = 0.5

    diff_mag_medians = 0. * diff_mag_median_bin_cents
    diff_mag_MADs = 0. * diff_mag_median_bin_cents

    diff_mag_MAD_hi = 0. * diff_mag_median_bin_cents
    diff_mag_MAD_lo = 0. * diff_mag_median_bin_cents

    for (cur_bin_cent,
         cur_bin_index) in zip(diff_mag_median_bin_cents,
                               range(len(diff_mag_medians))):
        cur_bin_hi = cur_bin_cent + (diff_mag_median_bin_size/2.)
        cur_bin_lo = cur_bin_cent - (diff_mag_median_bin_size/2.)
    
        mag_bin_filter = np.where(
            np.logical_and(common_detections_mag_table[legacy_version_str + '_mag'] >= cur_bin_lo,
                           common_detections_mag_table[legacy_version_str + '_mag'] < cur_bin_hi))
    
        filtered_diff_mags = diff_mag[mag_bin_filter]
    
        diff_mag_medians[cur_bin_index] = np.median(filtered_diff_mags)
        diff_mag_MADs[cur_bin_index] = stats.median_abs_deviation(filtered_diff_mags)
    
        diff_mag_MAD_hi[cur_bin_index] = (diff_mag_medians[cur_bin_index] +
                                          diff_mag_MADs[cur_bin_index])
        diff_mag_MAD_lo[cur_bin_index] = (diff_mag_medians[cur_bin_index] -
                                          diff_mag_MADs[cur_bin_index])

    # Plot luminosity functions for each mode
    mag_hist_bins = np.arange(9.0, 22.5, 0.5)
    if filt_name == 'h':
        mag_hist_bins = mag_hist_bins + 2.0
    
    plt.style.use(['presentation', 'ticks_outtie'])
    fig, ax = plt.subplots(figsize=(6,6))

    ax.set_title(epoch_name.replace('_', '\_') + ' ' + filt_label_strs[filt_name])
    
    ax.hist(stf_legacy_lis_table['m'], bins=mag_hist_bins,
            histtype='step', color='C0', lw=1.5,
            label=r"Legacy")
    
    ax.hist(stf_singPSF_lis_table['m'], bins=mag_hist_bins,
            histtype='step', color='C1',
            label=r"Single-PSF")
    
    ax.legend(loc='upper left')
    
    ax.set_xlabel(filt_mag_label_strs[filt_name])
    ax.set_ylabel(r"Stars Detected")
    
    x_majorLocator = MultipleLocator(2)
    x_minorLocator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(50)
    y_minorLocator = MultipleLocator(10)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)

    ax.set_xlim([np.min(mag_hist_bins), np.max(mag_hist_bins)])
    ax.set_ylim([0, 300])
    
    # Save out and close figure
    fig.tight_layout()
    
    fig.savefig(plot_out_dir + 'stf_lum_funcs.pdf')
    fig.savefig(plot_out_dir + 'stf_lum_funcs.png', dpi=200)

    plt.close(fig)
    
    
    # Plot luminosity function with one mode detections    
    plt.style.use(['presentation', 'ticks_outtie'])
    fig, ax = plt.subplots(figsize=(6,6))

    ax.set_title(epoch_name.replace('_', '\_') + ' ' + filt_label_strs[filt_name])
    
    ax.hist(legonly_detections_mag_table[legacy_version_str + '_mag'],
            bins=mag_hist_bins,
            histtype='step', color='C0', lw=1.5,
            label=r"Legacy Only")
    
    ax.hist(singonly_detections_mag_table[single_version_str + '_mag'],
            bins=mag_hist_bins,
            histtype='step', color='C1',
            label=r"Single-PSF Only")
    
    ax.legend(loc='upper left')
    
    ax.set_xlabel(filt_mag_label_strs[filt_name])
    ax.set_ylabel(r"Stars Detected")
    
    x_majorLocator = MultipleLocator(2)
    x_minorLocator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(20)
    y_minorLocator = MultipleLocator(5)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)

    ax.set_xlim([np.min(mag_hist_bins), np.max(mag_hist_bins)])
    ax.set_ylim([0, 150])
    
    # Save out and close figure
    fig.tight_layout()
    
    fig.savefig(plot_out_dir + 'stf_lum_funcs_one_mode.pdf')
    fig.savefig(plot_out_dir + 'stf_lum_funcs_one_mode.png', dpi=200)

    plt.close(fig)
    
    
    # Plot magnitude comparison
    fig, ax = plt.subplots(figsize=(6,6))
    
    ax.set_title(epoch_name.replace('_', '\_') + ' ' + filt_label_strs[filt_name])
    
    ax.plot(common_detections_mag_table[legacy_version_str + '_mag'],
            common_detections_mag_table[single_version_str + '_mag'],
            '.', color='royalblue', alpha=0.6)

    # Diagonal line for comparison
    ax.plot([9, 21], [9, 21], 'k--', lw=0.5)
    
    
    ax.set_xlabel(r"Legacy Starfinder: $m_{K'}$")
    ax.set_ylabel(r"Single PSF Starfinder: $m_{K'}$")


    x_majorLocator = MultipleLocator(2)
    x_minorLocator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(2)
    y_minorLocator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)

    ax.set_xlim([9, 21])
    ax.set_ylim([9, 21])

    ax.set_aspect('equal', 'box')
    ax.invert_yaxis()

    # Save out and close figure
    fig.tight_layout()
    
    fig.savefig(plot_out_dir + 'stf_mag_comparison.pdf')
    fig.savefig(plot_out_dir + 'stf_mag_comparison.png', dpi=200)

    plt.close(fig)
    

    # Plot delta magnitude comparison
    fig, ax = plt.subplots(figsize=(6,6))
    
    ax.set_title(epoch_name.replace('_', '\_') + ' ' + filt_label_strs[filt_name])
    
    ax.plot(common_detections_mag_table[legacy_version_str + '_mag'], diff_mag,
            '.', color='royalblue', alpha=0.6)

    # Flat and binned median lines for comparison
    ax.fill_between(diff_mag_median_bin_cents,
                    y1=diff_mag_MAD_lo, y2=diff_mag_MAD_hi,
                    facecolor='r', edgecolor='none', alpha=0.5)
    ax.plot(diff_mag_median_bin_cents, diff_mag_medians, 'r--', lw=0.5)


    ax.plot([9, 21], [0, 0], 'k--', lw=0.5)


    ax.set_xlabel(r"Legacy $m_{K'}$")
    ax.set_ylabel(r"Single PSF $m_{K'}$ $-$ Legacy $m_{K'}$")

    ax.text(10, -1.0, 'Brighter in Single PSF Starfinder',
            fontsize='x-small', va='center')
    ax.text(10, 1.0, 'Fainter in Single PSF Starfinder',
            fontsize='x-small', va='center')

    x_majorLocator = MultipleLocator(2)
    x_minorLocator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(0.5)
    y_minorLocator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)

    ax.set_xlim([9, 21])
    ax.set_ylim([-1.5, 1.5])

    # ax.set_aspect('equal', 'box')
    ax.invert_yaxis()

    # Save out and close figure
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'stf_mag_delta_comparison.pdf')
    fig.savefig(plot_out_dir + 'stf_mag_delta_comparison.png', dpi=200)

    plt.close(fig)


    # Plot delta magnitude comparison
    fig, ax = plt.subplots(figsize=(6,6))
    
    ax.set_title(epoch_name.replace('_', '\_') + ' ' + filt_label_strs[filt_name])
    
    # ax.plot(common_detections_mag_table['v2_1_mag'], diff_mag,
    #         '.', color='royalblue', alpha=0.6)

    # Flat and binned median lines for comparison
    ax.fill_between(diff_mag_median_bin_cents,
                    y1=diff_mag_MAD_lo, y2=diff_mag_MAD_hi,
                    facecolor='r', edgecolor='none', alpha=0.5)
    ax.plot(diff_mag_median_bin_cents, diff_mag_medians, 'r--', lw=0.5)


    ax.plot([9, 21], [0, 0], 'k--', lw=0.5)


    ax.set_xlabel(r"Legacy $m_{K'}$")
    ax.set_ylabel(r"Single PSF $m_{K'}$ $-$ Legacy $m_{K'}$")

    ax.text(10, -1.0, 'Brighter in Single PSF Starfinder',
            fontsize='x-small', va='center')
    ax.text(10, 1.0, 'Fainter in Single PSF Starfinder',
            fontsize='x-small', va='center')

    x_majorLocator = MultipleLocator(2)
    x_minorLocator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(0.5)
    y_minorLocator = MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)

    ax.set_xlim([9, 21])
    ax.set_ylim([-1.5, 1.5])

    # ax.set_aspect('equal', 'box')
    ax.invert_yaxis()

    # Save out and close figure
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'stf_mag_delta_comparison_nostars.pdf')
    fig.savefig(plot_out_dir + 'stf_mag_delta_comparison_nostars.png', dpi=200)

    plt.close(fig)

# Read in epochs table
epochs_table = Table.read('epochs_table.h5', format='hdf5', path='data')

epochs_table = epochs_table[np.where(epochs_table['nights_combo'] == 'single_night')]

# Run analysis code on all epochs
for epochs_row in tqdm(epochs_table):
    cur_epoch = epochs_row['epoch']
    cur_filt = epochs_row['filt']
    
    analyze_mag_comparison(cur_epoch, dr_path = dr_path,
                           filt_name = cur_filt,
                           legacy_version_str = legacy_version_str,
                           single_version_str = single_version_str)
    
