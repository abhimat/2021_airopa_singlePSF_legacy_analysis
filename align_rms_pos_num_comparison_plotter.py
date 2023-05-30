#!/usr/bin/env python

# Starlist magnitude and position comparison on the field
# ---
# Abhimat Gautam

import os

import numpy as np
from scipy import stats
from scipy.spatial import KDTree

from file_readers import stf_rms_lis_reader,\
    align_orig_pos_reader, align_pos_reader, align_pos_err_reader,\
    align_mag_reader, align_param_reader

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
                           filt_name = 'kp',
                           legacy_version_str = 'v2_3',
                           single_version_str = 'v3_1',
                           mag_bin_lo = -1, mag_bin_hi = -1,
                           num_near_neighbors=20,
                          ):
    cur_wd = os.getcwd()
    
    epoch_analysis_location = '{0}/{1}_{2}/'.format(cur_wd, epoch_name, filt_name)
    starlist_align_location = epoch_analysis_location + 'starlists_align_rms/align/'
    align_root = starlist_align_location + 'align_d_rms_abs'
    
    plot_out_dir = epoch_analysis_location + 'align_rms_pos_num_comparison_plots/'
    os.makedirs(plot_out_dir, exist_ok=True)
    
    out_mag_suffix = ''
    if mag_bin_lo != -1:
        out_mag_suffix = f'_mag_{mag_bin_lo}_{mag_bin_hi}'
    
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
    if not os.path.exists(align_root + '.mag'):
        print(f"Align RMS for {epoch_name} in {filt_name} not completed")
        return
    
    align_mag_table = align_mag_reader(align_root + '.mag',
                                       align_stf_1_version=legacy_version_str,
                                       align_stf_2_version=single_version_str)

    align_pos_table = align_pos_reader(align_root + '.pos',
                                       align_stf_1_version=legacy_version_str,
                                       align_stf_2_version=single_version_str)
    
    align_pos_err_table = align_pos_err_reader(align_root + '.err',
                              align_stf_1_version=legacy_version_str,
                              align_stf_2_version=single_version_str)
    
    align_orig_pos_table = align_orig_pos_reader(align_root + '.origpos',
                                                 align_stf_1_version=legacy_version_str,
                                                 align_stf_2_version=single_version_str)
    
    align_param_table = align_param_reader(
        align_root + '.param',
        align_stf_1_version=legacy_version_str,
        align_stf_2_version=single_version_str)
    
    # Common detections in each starlist
    
    
    det_both = np.logical_and(align_mag_table[legacy_version_str + '_mag'] != 0.0,
                       align_mag_table[single_version_str + '_mag'] != 0.0)
    
    detection_filter = np.where(det_both)
    
    if mag_bin_lo != -1:
        det_mag = np.logical_and(align_mag_table[single_version_str + '_mag'] >= mag_bin_lo,
                       align_mag_table[single_version_str + '_mag'] < mag_bin_hi)
        
        detection_filter = np.where(np.logical_and(det_both, det_mag))
        
        # print(detection_filter)
    
    common_detections_mag_table = align_mag_table[detection_filter]
    common_detections_pos_table = align_pos_table[detection_filter]
    common_detections_pos_err_table = align_pos_err_table[detection_filter]
    common_detections_orig_pos_table = align_orig_pos_table[detection_filter]
    common_detections_param_table = align_param_table[detection_filter]
    
    common_detections_r2d_table = np.hypot(
        common_detections_pos_table[legacy_version_str + '_x'],
        common_detections_pos_table[legacy_version_str + '_y'])
    
    # One mode detection
    leg_detect = np.logical_and(align_mag_table[legacy_version_str + '_mag'] != 0.0,
                       align_mag_table[single_version_str + '_mag'] == 0.0)
    
    leg_detection_filter = np.where(leg_detect)
    
    sin_detect = np.logical_and(align_mag_table[legacy_version_str + '_mag'] == 0.0,
                       align_mag_table[single_version_str + '_mag'] != 0.0)
    
    sin_detection_filter = np.where(sin_detect)
    
    if mag_bin_lo != -1:
        leg_det_mag = np.logical_and(align_mag_table[legacy_version_str + '_mag'] >= mag_bin_lo,
                       align_mag_table[legacy_version_str + '_mag'] < mag_bin_hi)
        
        leg_detection_filter = np.where(np.logical_and(leg_detect, leg_det_mag))
        
        sin_det_mag = np.logical_and(align_mag_table[single_version_str + '_mag'] >= mag_bin_lo,
                       align_mag_table[single_version_str + '_mag'] < mag_bin_hi)
        
        sin_detection_filter = np.where(np.logical_and(sin_detect, sin_det_mag))
    
    legonly_detections_mag_table = align_mag_table[leg_detection_filter]
    legonly_detections_pos_table = align_pos_table[leg_detection_filter]
    legonly_detections_pos_err_table = align_pos_err_table[leg_detection_filter]
    legonly_detections_orig_pos_table = align_orig_pos_table[leg_detection_filter]
    legonly_detections_param_table = align_param_table[leg_detection_filter]
    
    legonly_detections_r2d_table = np.hypot(
        legonly_detections_pos_table[legacy_version_str + '_x'],
        legonly_detections_pos_table[legacy_version_str + '_y'])
    
    sinonly_detections_mag_table = align_mag_table[sin_detection_filter]
    sinonly_detections_pos_table = align_pos_table[sin_detection_filter]
    sinonly_detections_pos_err_table = align_pos_err_table[sin_detection_filter]
    sinonly_detections_orig_pos_table = align_orig_pos_table[sin_detection_filter]
    sinonly_detections_param_table = align_param_table[sin_detection_filter]
    
    sinonly_detections_r2d_table = np.hypot(
        sinonly_detections_pos_table[single_version_str + '_x'],
        sinonly_detections_pos_table[single_version_str + '_y'])
    
    # Draw r2d histogram
    plt.style.use(['ticks_outtie'])
    
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    hist_bins = np.arange(0, 5, 0.75)
    
    (common_hist_r2d, common_bins) = np.histogram(
        common_detections_r2d_table,
        bins=hist_bins)
    (legonly_hist_r2d, legonly_bins) = np.histogram(
        legonly_detections_r2d_table,
        bins=hist_bins
    )
    (sinonly_hist_r2d, sinonly_bins) = np.histogram(
        sinonly_detections_r2d_table,
        bins=hist_bins
    )
    
    # ax.hist(common_detections_r2d_table,
    #         bins=hist_bins,
    #         histtype='step',
    #         color='k', label='Detections in both modes')
    
    ax.hist(hist_bins[:-1],
            bins=hist_bins,
            weights=legonly_hist_r2d/common_hist_r2d,
            histtype='step',
            color='C0', label='Legacy only detections')
    
    ax.hist(hist_bins[:-1],
            bins=hist_bins,
            weights=sinonly_hist_r2d/common_hist_r2d,
            histtype='step',
            color='C1', label='Single-PSF only detections')
    
    ax.set_xlabel(r'Distance from Sgr A* (arcsec)')
    ax.set_ylabel('Unique Dets. / Common Dets.')
    
    ax.legend(loc='upper right')
    
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 0.5])
    
    x_majorLocator = MultipleLocator(1)
    x_minorLocator = MultipleLocator(0.25)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(.1)
    y_minorLocator = MultipleLocator(.02)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'pos_num_hist.pdf')
    fig.savefig(plot_out_dir + 'pos_num_hist.png', dpi=200)

    plt.close(fig)
    
    # Draw mag histogram
    plt.style.use(['ticks_outtie'])
    
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    hist_bins = np.arange(9, 21, 0.5)
    
    (common_hist_mag, common_bins) = np.histogram(
        common_detections_mag_table[single_version_str + '_mag'],
        bins=hist_bins)
    (legonly_hist_mag, legonly_bins) = np.histogram(
        legonly_detections_mag_table[legacy_version_str + '_mag'],
        bins=hist_bins
    )
    (sinonly_hist_mag, sinonly_bins) = np.histogram(
        sinonly_detections_mag_table[single_version_str + '_mag'],
        bins=hist_bins
    )
    
    # Write out table for the central arcsecond histogram
    mag_hist_table = Table(
        [
            hist_bins[:-1],
            hist_bins[1:],
            common_hist_mag,
            legonly_hist_mag,
            sinonly_hist_mag,
        ],
        names=(
            'mag_bin_lo',
            'mag_bin_hi',
            'num_dets_common',
            'num_dets_legonly',
            'num_dets_sinonly',
        )
    )
    
    mag_hist_table.write(
        plot_out_dir + 'mag_num_hist.txt',
        format='ascii.fixed_width_two_line',
        overwrite=True,
    )
    mag_hist_table.write(
        plot_out_dir + 'mag_num_hist.h5',
        format='hdf5', path='data', serialize_meta=True,
        overwrite=True,
    )
    
    # Draw plot
    ax.hist(
        hist_bins[:-1],
        bins=hist_bins,
        weights=common_hist_mag,
        histtype='step',
        color='k', label='Detections in both modes',
    )
    
    ax.hist(
        hist_bins[:-1],
        bins=hist_bins,
        weights=legonly_hist_mag,
        histtype='step',
        hatch='///',
        color='C0', label='Legacy only detections',
    )
    
    ax.hist(
        hist_bins[:-1],
        bins=hist_bins,
        weights=sinonly_hist_mag,
        histtype='step',
        hatch='\\\\\\',
        color='C1', label='Single-PSF only detections',
    )
    
    ax.set_xlabel(r'Mag')
    ax.set_ylabel('Detections')
    
    ax.legend(loc='upper left')
    
    ax.set_xlim([9.5, 20.5])
    # ax.set_ylim([0, 0.5])
    
    x_majorLocator = MultipleLocator(2)
    x_minorLocator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(50)
    y_minorLocator = MultipleLocator(10)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    ax.set_title(epoch_name.replace('_', '\_'))
    
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'mag_num_hist.pdf')
    fig.savefig(plot_out_dir + 'mag_num_hist.png', dpi=200)

    plt.close(fig)
    
    # Draw mag histogram for just the central arcsecond
    plt.style.use(['ticks_outtie'])
    
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    hist_bins = np.arange(9, 21, 0.5)
    
    (common_hist_mag, common_bins) = np.histogram(
        (common_detections_mag_table[single_version_str + '_mag'])[
            np.where(common_detections_r2d_table <= 1.0)
        ],
        bins=hist_bins)
    (legonly_hist_mag, legonly_bins) = np.histogram(
        (legonly_detections_mag_table[legacy_version_str + '_mag'])[
            np.where(legonly_detections_r2d_table <= 1.0)
        ],
        bins=hist_bins
    )
    (sinonly_hist_mag, sinonly_bins) = np.histogram(
        (sinonly_detections_mag_table[single_version_str + '_mag'])[
            np.where(sinonly_detections_r2d_table <= 1.0)
        ],
        bins=hist_bins
    )
    
    # Write out table for the central arcsecond histogram
    mag_hist_table = Table(
        [
            hist_bins[:-1],
            hist_bins[1:],
            common_hist_mag,
            legonly_hist_mag,
            sinonly_hist_mag,
        ],
        names=(
            'mag_bin_lo',
            'mag_bin_hi',
            'num_dets_common',
            'num_dets_legonly',
            'num_dets_sinonly',
        )
    )
    
    mag_hist_table.write(
        plot_out_dir + 'mag_num_hist_centarcsec.txt',
        format='ascii.fixed_width_two_line',
        overwrite=True,
    )
    mag_hist_table.write(
        plot_out_dir + 'mag_num_hist_centarcsec.h5',
        format='hdf5', path='data', serialize_meta=True,
        overwrite=True,
    )
    
    # Draw plot
    ax.hist(
        hist_bins[:-1],
        bins=hist_bins,
        weights=common_hist_mag,
        histtype='step',
        color='k', label='Detections in both modes',
    )
    
    ax.hist(
        hist_bins[:-1],
        bins=hist_bins,
        weights=legonly_hist_mag,
        histtype='step',
        hatch='///',
        color='C0', label='Legacy only detections',
    )
    
    ax.hist(
        hist_bins[:-1],
        bins=hist_bins,
        weights=sinonly_hist_mag,
        histtype='step',
        hatch='\\\\\\',
        color='C1', label='Single-PSF only detections',
    )
    
    ax.set_xlabel(r'Mag')
    ax.set_ylabel('Central Arcsecond Detections')
    
    ax.legend(loc='upper left')
    
    ax.set_xlim([9.5, 20.5])
    # ax.set_ylim([0, 0.5])
    
    x_majorLocator = MultipleLocator(2)
    x_minorLocator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(5)
    y_minorLocator = MultipleLocator(1)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    ax.set_title(epoch_name.replace('_', '\_'))
    
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'mag_num_hist_centarcsec.pdf')
    fig.savefig(plot_out_dir + 'mag_num_hist_centarcsec.png', dpi=200)

    plt.close(fig)
    
    # Draw corr comparison
    plt.style.use(['ticks_outtie'])
    
    fig, ax = plt.subplots(figsize=(6,6), frameon=False)
    
    ax.plot(common_detections_param_table[legacy_version_str + '_corr'],
            common_detections_param_table[single_version_str + '_corr'],
            'ko', alpha=0.2)
    
    ax.set_xlabel(r'Legacy Correlation')
    ax.set_ylabel(r'Single-PSF Correlation')
    
    # ax.legend(loc='upper left')
    
    # ax.set_xlim([9.5, 20.5])
    # ax.set_ylim([0, 0.5])
    
    # x_majorLocator = MultipleLocator(2)
    # x_minorLocator = MultipleLocator(0.5)
    # ax.xaxis.set_major_locator(x_majorLocator)
    # ax.xaxis.set_minor_locator(x_minorLocator)
    #
    # y_majorLocator = MultipleLocator(50)
    # y_minorLocator = MultipleLocator(10)
    # ax.yaxis.set_major_locator(y_majorLocator)
    # ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'corr_comparison.pdf')
    fig.savefig(plot_out_dir + 'corr_comparison.png', dpi=200)

    plt.close(fig)
    
    # Draw common corr histogram
    plt.style.use(['ticks_outtie'])
    
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    hist_bins = np.linspace(0.8, 1.0, num=11)
    
    # (common_hist_mag, common_bins) = np.histogram(
    #     common_detections_mag_table[single_version_str + '_mag'],
    #     bins=hist_bins)
    # (legonly_hist_mag, legonly_bins) = np.histogram(
    #     legonly_detections_mag_table[legacy_version_str + '_mag'],
    #     bins=hist_bins
    # )
    # (sinonly_hist_mag, sinonly_bins) = np.histogram(
    #     sinonly_detections_mag_table[single_version_str + '_mag'],
    #     bins=hist_bins
    # )
    #
    # print(common_hist_mag)
    # print(legonly_hist_mag)
    
    # ax.hist(hist_bins[:-1],
    #         bins=hist_bins,
    #         weights=common_hist_mag,
    #         histtype='step',
    #         color='k', label='Detections in both modes')
    
    ax.hist(common_detections_param_table[legacy_version_str + '_corr'],
            bins=hist_bins,
            # weights=legonly_hist_mag,
            histtype='step',
            color='C0', label='Common detections, legacy mode correlation')
    
    ax.hist(common_detections_param_table[single_version_str + '_corr'],
            bins=hist_bins,
            # weights=legonly_hist_mag,
            histtype='step',
            color='C1', label='Common detections, single-PSF mode correlation')
    
    ax.set_xlabel(r'Correlation')
    ax.set_ylabel('Detections')
    
    ax.legend(loc='upper left')
    
    # ax.set_xlim([9.5, 20.5])
    # ax.set_ylim([0, 0.5])
    
    # x_majorLocator = MultipleLocator(2)
    # x_minorLocator = MultipleLocator(0.5)
    # ax.xaxis.set_major_locator(x_majorLocator)
    # ax.xaxis.set_minor_locator(x_minorLocator)
    #
    # y_majorLocator = MultipleLocator(50)
    # y_minorLocator = MultipleLocator(10)
    # ax.yaxis.set_major_locator(y_majorLocator)
    # ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'common_corr_hist.pdf')
    fig.savefig(plot_out_dir + 'common_corr_hist.png', dpi=200)

    plt.close(fig)
    
    
    # Draw unique corr histogram
    plt.style.use(['ticks_outtie'])
    
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    hist_bins = np.linspace(0.8, 1.0, num=11)
    
    # (common_hist_mag, common_bins) = np.histogram(
    #     common_detections_mag_table[single_version_str + '_mag'],
    #     bins=hist_bins)
    # (legonly_hist_mag, legonly_bins) = np.histogram(
    #     legonly_detections_mag_table[legacy_version_str + '_mag'],
    #     bins=hist_bins
    # )
    # (sinonly_hist_mag, sinonly_bins) = np.histogram(
    #     sinonly_detections_mag_table[single_version_str + '_mag'],
    #     bins=hist_bins
    # )
    #
    # print(common_hist_mag)
    # print(legonly_hist_mag)
    
    # ax.hist(hist_bins[:-1],
    #         bins=hist_bins,
    #         weights=common_hist_mag,
    #         histtype='step',
    #         color='k', label='Detections in both modes')
    
    ax.hist(legonly_detections_param_table[legacy_version_str + '_corr'],
            bins=hist_bins,
            # weights=legonly_hist_mag,
            histtype='step',
            color='C0', label='Unique detections, legacy mode correlation')
    
    ax.hist(sinonly_detections_param_table[single_version_str + '_corr'],
            bins=hist_bins,
            # weights=legonly_hist_mag,
            histtype='step',
            color='C1', label='Unique detections, single-PSF mode correlation')
    
    ax.set_xlabel(r'Correlation')
    ax.set_ylabel('Detections')
    
    ax.legend(loc='upper left')
    
    # ax.set_xlim([9.5, 20.5])
    # ax.set_ylim([0, 0.5])
    
    # x_majorLocator = MultipleLocator(2)
    # x_minorLocator = MultipleLocator(0.5)
    # ax.xaxis.set_major_locator(x_majorLocator)
    # ax.xaxis.set_minor_locator(x_minorLocator)
    #
    # y_majorLocator = MultipleLocator(50)
    # y_minorLocator = MultipleLocator(10)
    # ax.yaxis.set_major_locator(y_majorLocator)
    # ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'unique_corr_hist.pdf')
    fig.savefig(plot_out_dir + 'unique_corr_hist.png', dpi=200)

    plt.close(fig)
    
    # Draw all corr histogram
    plt.style.use(['ticks_outtie'])
    
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    hist_bins = np.arange(0.8, 1.2, 0.02)
    
    (common_hist_corr, common_bins) = np.histogram(
        (common_detections_param_table[legacy_version_str + '_corr'] +\
        common_detections_param_table[single_version_str + '_corr'])/2.,
        bins=hist_bins)
    (legonly_hist_corr, legonly_bins) = np.histogram(
        legonly_detections_param_table[legacy_version_str + '_corr'],
        bins=hist_bins
    )
    (sinonly_hist_corr, sinonly_bins) = np.histogram(
        sinonly_detections_param_table[single_version_str + '_corr'],
        bins=hist_bins
    )
    #
    # print(common_hist_mag)
    # print(legonly_hist_mag)
    
    ax.hist(hist_bins[:-1],
            bins=hist_bins,
            weights=common_hist_corr,
            histtype='step',
            color='k', label='Detections in both modes')
    
    ax.hist(hist_bins[:-1],
            bins=hist_bins,
            weights=legonly_hist_corr,
            histtype='step',
            color='C0', label='Legacy only detection')
    
    ax.hist(hist_bins[:-1],
            bins=hist_bins,
            weights=sinonly_hist_corr,
            histtype='step',
            color='C1', label='Single-PSF only detection')
    
    ax.set_xlabel(r'Correlation')
    ax.set_ylabel(r'Detections')
    
    ax.legend(loc='upper left')
    
    ax.set_xlim([0.8, 1.0])
    # ax.set_ylim([0, 0.5])
    
    # x_majorLocator = MultipleLocator(2)
    # x_minorLocator = MultipleLocator(0.5)
    # ax.xaxis.set_major_locator(x_majorLocator)
    # ax.xaxis.set_minor_locator(x_minorLocator)
    #
    # y_majorLocator = MultipleLocator(50)
    # y_minorLocator = MultipleLocator(10)
    # ax.yaxis.set_major_locator(y_majorLocator)
    # ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'corr_hist.pdf')
    fig.savefig(plot_out_dir + 'corr_hist.png', dpi=200)

    plt.close(fig)
    
    # Draw common correlation ratio vs. avg plot
    plt.style.use(['ticks_outtie'])
    
    common_corr_ratio =\
        common_detections_param_table[single_version_str + '_corr'] /\
        common_detections_param_table[legacy_version_str + '_corr']
    common_corr_avg =\
        (common_detections_param_table[single_version_str + '_corr'] +
         common_detections_param_table[legacy_version_str + '_corr'])/2.
    
    median_ratio = np.median(common_corr_ratio)
    std_sqrt_ratio = np.std(common_corr_ratio) / np.sqrt(len(common_corr_ratio))
    
    print(median_ratio)
    print(std_sqrt_ratio)
    
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    ax.plot(common_corr_avg,
            common_corr_ratio,
            'k.', alpha=0.2)
    
    # ax.axhline(1.0, color='k', ls='--', lw=0.5, label='Equal Corr')
    
    ax.axhline(median_ratio, color='C4', ls='-', lw=1.0,
               label='Median Ratio')
    
    ax.fill_between(
        [0.75, 1.05],
        y1=[median_ratio + std_sqrt_ratio],
        y2=[median_ratio - std_sqrt_ratio],
        color='C4')
    
    # ax.axhline(std_sqrt_ratio, color='C5', ls=':', lw=0.5,
    #            label=r'Std. Dev. Ratio / $\sqrt{N}$$')
    
    ax.set_xlabel('Avg. Detection Correlation')
    ax.set_ylabel('Single-PSF Corr. / Legacy Corr.')
    
    ax.legend(loc='lower left')
    
    ax.set_xlim([0.78, 1.02])
    ax.set_ylim([0.75, 1.25])
    
    x_majorLocator = MultipleLocator(0.025)
    x_minorLocator = MultipleLocator(0.005)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(0.1)
    y_minorLocator = MultipleLocator(0.02)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'common_corr_ratio_avg.pdf')
    fig.savefig(plot_out_dir + 'common_corr_ratio_avg.png', dpi=200)

    plt.close(fig)
    
    # Draw common correlation ratio vs. r2d    
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    ax.plot(common_detections_r2d_table,
            common_corr_ratio,
            'k.', alpha=0.2)
    
    # ax.axhline(1.0, color='k', ls='--', lw=0.5, label='Equal Corr')
    
    # ax.axhline(median_ratio, color='C4', ls='-', lw=1.0,
    #            label='Median Ratio')
    #
    # ax.fill_between(
    #     [0.75, 1.05],
    #     y1=[median_ratio + std_sqrt_ratio],
    #     y2=[median_ratio - std_sqrt_ratio],
    #     color='C4')
    
    # ax.axhline(std_sqrt_ratio, color='C5', ls=':', lw=0.5,
    #            label=r'Std. Dev. Ratio / $\sqrt{N}$$')
    
    ax.set_xlabel('Distance from Sgr A* (arcsec)')
    ax.set_ylabel('Single-PSF Corr. / Legacy Corr.')
    
    # ax.legend(loc='lower left')
    
    ax.set_xlim([0, 7.5])
    ax.set_ylim([0.75, 1.25])
    
    x_majorLocator = MultipleLocator(2.0)
    x_minorLocator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(0.1)
    y_minorLocator = MultipleLocator(0.02)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'common_corr_ratio_r2d.pdf')
    fig.savefig(plot_out_dir + 'common_corr_ratio_r2d.png', dpi=200)

    plt.close(fig)
    
    # Draw common correlation ratio vs. r2d    
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    ax.plot(common_detections_r2d_table,
            common_corr_ratio,
            'k.', alpha=0.4)
    
    central_half_arcsec_filt = np.where(common_detections_r2d_table < 0.5)
    central_half_arcsec_cut = common_corr_ratio[central_half_arcsec_filt]
    
    central_half_arcsec_median_ratio = np.median(central_half_arcsec_cut)
    central_half_arcsec_std_sqrt_ratio = np.std(central_half_arcsec_cut) / \
        np.sqrt(len(central_half_arcsec_cut))
    
    # ax.axhline(1.0, color='k', ls='--', lw=0.5, label='Equal Corr')
    
    ax.axhline(median_ratio, color='C4', ls='-', lw=1.0,
               label='Median ratio, all stars')
    ax.fill_between(
        [0, 7.5],
        y1=[median_ratio + std_sqrt_ratio],
        y2=[median_ratio - std_sqrt_ratio],
        color='C4')
    
    ax.plot([0, 0.5],
            [central_half_arcsec_median_ratio, central_half_arcsec_median_ratio],
            color='C5', ls='-', lw=1.0, alpha=0.5,
            label='Median ratio, central half arcsecond stars')
    ax.fill_between(
        [0, 0.5],
        y1=[central_half_arcsec_median_ratio + central_half_arcsec_std_sqrt_ratio],
        y2=[central_half_arcsec_median_ratio - central_half_arcsec_std_sqrt_ratio],
        color='C5', alpha=0.5)
    
    # ax.axhline(std_sqrt_ratio, color='C5', ls=':', lw=0.5,
    #            label=r'Std. Dev. Ratio / $\sqrt{N}$$')
    
    ax.set_xlabel('Distance from Sgr A* (arcsec)')
    ax.set_ylabel('Single-PSF Corr. / Legacy Corr.')
    
    ax.legend(loc='lower right')
    
    ax.set_xlim([0, 2.0])
    ax.set_ylim([0.75, 1.25])
    
    x_majorLocator = MultipleLocator(0.5)
    x_minorLocator = MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(0.1)
    y_minorLocator = MultipleLocator(0.02)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'common_corr_ratio_r2d_central_half_arcsec.pdf')
    fig.savefig(plot_out_dir + 'common_corr_ratio_r2d_central_half_arcsec.png', dpi=200)

    plt.close(fig)
    


# Read in epochs table
epochs_table = Table.read('epochs_table.h5', format='hdf5', path='data')

# print(epochs_table)

# epochs_table = epochs_table[np.where(epochs_table['nights_combo'] == 'single_night')]

# print(epochs_table)

# Run analysis code on all epoch

# analyze_pos_comparison(
#     '20160503nirc2', dr_path = dr_path,
#     filt_name = 'kp',
#     legacy_version_str = legacy_version_str,
#     single_version_str = single_version_str,
# )

# epochs_list = list(range(0,61)) + list(range(62, 91)) + list(range(92, 118))
# epochs_list = list(range(123, 123))

for epochs_row in tqdm(epochs_table):
    cur_epoch = epochs_row['epoch']
    print(cur_epoch)
    cur_filt = epochs_row['filt']

    analyze_pos_comparison(cur_epoch, dr_path = dr_path,
                           filt_name = cur_filt,
                           legacy_version_str = legacy_version_str,
                           single_version_str = single_version_str)


    # # Run this comparison in different mag bins
    # mag_bins_lo = [11, 13, 15, 17]
    # mag_bins_hi = [13, 15, 17, 25]
    #
    # num_near_neighbors_vals = [5, 10, 20, 10]
    #
    # if cur_filt == 'h':
    #     mag_bins_lo = [13, 15, 17, 19]
    #     mag_bins_hi = [15, 17, 19, 27]
    #
    # for (cur_bin_lo, cur_bin_hi,
    #      num_near_neighbors,
    #     ) in zip(mag_bins_lo, mag_bins_hi,
    #              num_near_neighbors_vals):
    #     analyze_pos_comparison(cur_epoch, dr_path = dr_path,
    #                            filt_name = cur_filt,
    #                            legacy_version_str = legacy_version_str,
    #                            single_version_str = single_version_str,
    #                            mag_bin_lo = cur_bin_lo, mag_bin_hi = cur_bin_hi,
    #                            num_near_neighbors = num_near_neighbors,
    #                           )


