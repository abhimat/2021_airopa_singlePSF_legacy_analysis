#!/usr/bin/env python

# Starlist: magnitude and number comparison
# ---
# Abhimat Gautam

import numpy as np

from file_readers import stf_lis_reader,\
    align_orig_pos_reader, align_pos_reader, align_mag_reader

from astropy.table import Table
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator

import os
from tqdm import tqdm
import warnings

dr_path = '/g/ghez/data/dr/dr1'
legacy_version_str = 'v2_3'
single_version_str = 'v3_1'
stf_corr = '0.8'

plate_scale = 0.00993   ## NIRC2 Plate Scale

def resid_hist_plotter(epoch_name, dr_path = '/g/ghez/data/dr/dr1',
                       filt_name = 'kp', stf_corr = '0.8',
                       legacy_version_str = 'v2_3',
                       single_version_str = 'v3_1'):
    cur_wd = os.getcwd()
    
    # Analysis Data Location
    epoch_analysis_location = '{0}/{1}_{2}/'.format(cur_wd, epoch_name, filt_name)
    starlist_align_location = epoch_analysis_location + 'starlists_align_rms/align/'
    align_root = starlist_align_location + 'align_d_rms'
    
    plot_out_dir = epoch_analysis_location + 'resid_stats_plots/'
    os.makedirs(plot_out_dir, exist_ok=True)
    
    stf_legacy_name = 'starfinder_' + legacy_version_str
    stf_singPSF_name = 'starfinder_' + single_version_str
    
    stf_res_versions = ['legacy', 'singPSF']
    
    # Read original image from FITS file
    image_data_orig = np.array([])
    
    image_fits_file = '{0}/combo/{1}/mag{1}_{2}.fits'.format(
        dr_path, epoch_name, filt_name)
    
    warnings.simplefilter('ignore', UserWarning)
    with fits.open(image_fits_file) as hdulist:
        image_data_orig = hdulist[0].data
    
    image_sig_data = np.array([])
    
    image_sig_fits_file = '{0}/combo/{1}/mag{1}_{2}_sig.fits'.format(
        dr_path, epoch_name, filt_name)
    
    warnings.simplefilter('ignore', UserWarning)
    with fits.open(image_sig_fits_file) as hdulist:
        image_sig_data = hdulist[0].data
    
    
    # Read in resid images' data
    resid_data_leg_orig = np.array([])
    resid_data_sin_orig = np.array([])

    for stf_res_version in stf_res_versions:
        if stf_res_version == 'legacy':
            stf_res_name = stf_legacy_name
        elif stf_res_version == 'singPSF':
            stf_res_name = stf_singPSF_name
        
        # Read resid image from the FITS file
        stf_res_fits_file = '{0}/starlists/combo/{1}/{2}/mag{1}_{3}_res.fits'.format(
            dr_path, epoch_name,
            stf_res_name, filt_name)
        
        if not os.path.exists(stf_res_fits_file):
            print(f"Residual file for {epoch_name} does not exist")
            return
        
        warnings.simplefilter('ignore', UserWarning)
        with fits.open(stf_res_fits_file) as hdulist:
            if stf_res_version == 'legacy':
                resid_data_leg_orig = hdulist[0].data
            elif stf_res_version == 'singPSF':
                resid_data_sin_orig = hdulist[0].data
    
    # Draw a 97% cut in sig to determine where on image to perform this analysis
    sig_max = np.max(image_sig_data)
    sig_cut = np.where(image_sig_data >= 0.97 * sig_max)
    
    # Implement sig cuts in image and resid data
    image_data = image_data_orig[sig_cut]
    resid_data_leg = resid_data_leg_orig[sig_cut]
    resid_data_sin = resid_data_sin_orig[sig_cut]
    
    total_pixels = len(image_data.flatten())
    print(f'Pixels used in analysis: {total_pixels} pixels')
    
    image_mask = np.where(image_data > 0)
    
    # Save out data
    out_table = Table(
        [
            [np.nanmedian(np.sqrt(resid_data_leg[image_mask].flatten() ** 2))],
            [np.nanmedian(np.sqrt(resid_data_sin[image_mask].flatten() ** 2))],
            [np.nanmedian(np.sqrt(resid_data_leg[image_mask].flatten()**2) / image_data[image_mask].flatten())],
            [np.nanmedian(np.sqrt(resid_data_sin[image_mask].flatten()**2) / image_data[image_mask].flatten())],
        ],
        names=(
            'resid_abs_leg',
            'resid_abs_sin',
            'resid_abs_feu_leg',
            'resid_abs_feu_sin',
        ),
    )
    
    out_table.write(
        plot_out_dir + 'resids.txt',
        format='ascii.fixed_width_two_line',
        overwrite=True,
    )
    out_table.write(
        plot_out_dir + 'resids.h5',
        format='hdf5', path='data', serialize_meta=True,
        overwrite=True,
    )
    
    # Draw histogram
    plt.style.use(['ticks_outtie'])
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    hist_bins = np.linspace(0, 1000, num=51)
    
    ax.hist(np.sqrt(resid_data_leg.flatten() ** 2), bins=hist_bins,
            histtype='step',
            color='C0', label='Legacy')
    
    ax.hist(np.sqrt(resid_data_sin.flatten() ** 2), bins=hist_bins,
            histtype='step',
            color='C1', label='Single-PSF')
    
    ax.axvline(np.nanmedian(np.sqrt(resid_data_leg.flatten()**2)),
               color='C0', ls='--')
    ax.axvline(np.nanmedian(np.sqrt(resid_data_sin.flatten()**2)),
               color='C1', ls='--')
    
    print(np.nanmedian(np.sqrt(resid_data_leg.flatten()**2)))
    print(np.nanmedian(np.sqrt(resid_data_sin.flatten()**2)))
    
    ax.set_xlabel(r'$\sqrt{(data - model)^2}$')
    ax.set_ylabel('Number of pixels')
    
    ax.set_xlim([0, 1000])
    
    ax.legend()
    
    x_majorLocator = MultipleLocator(200)
    x_minorLocator = MultipleLocator(50)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(2e4)
    y_minorLocator = MultipleLocator(5e3)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'resid_hist.pdf')
    fig.savefig(plot_out_dir + 'resid_hist.png', dpi=200)

    plt.close(fig)
    
    # Draw histogram
    plt.style.use(['ticks_outtie'])
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    hist_bins = np.linspace(0, 5, num=51)
    # hist_bins = np.linspace(0, 10, num=80)
    # hist_bins = np.logspace(-5, 2, num=100)
    
    
    ax.hist(np.sqrt(resid_data_leg[image_mask].flatten()**2) / image_data[image_mask].flatten(),
            bins=hist_bins,
            histtype='step',
            color='C0', label='Legacy')
    
    ax.hist(np.sqrt(resid_data_sin[image_mask].flatten()**2) / image_data[image_mask].flatten(),
            bins=hist_bins,
            histtype='step',
            color='C1', label='Single-PSF')
    
    print(np.nanmedian(np.sqrt(resid_data_leg.flatten()**2) / image_data.flatten()))
    print(np.nanmedian(np.sqrt(resid_data_sin.flatten()**2) / image_data.flatten()))
    
    ax.axvline(np.nanmedian(np.sqrt(resid_data_leg.flatten()**2) / image_data.flatten()),
               color='C0', ls='--')
    ax.axvline(np.nanmedian(np.sqrt(resid_data_sin.flatten()**2) / image_data.flatten()),
               color='C1', ls='--')
    
    ax.set_xlabel(r'$\sqrt{(data-model)^2}$ / $data$')
    ax.set_ylabel('Number of pixels')
    
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    ax.set_xlim([0, 5])
    
    # ax.legend(loc='upper left')
    ax.legend()
    
    x_majorLocator = MultipleLocator(1)
    x_minorLocator = MultipleLocator(0.2)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(5e4)
    y_minorLocator = MultipleLocator(1e4)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()

    fig.savefig(plot_out_dir + 'resid_feu_hist.pdf')
    fig.savefig(plot_out_dir + 'resid_feu_hist.png', dpi=200)

    plt.close(fig)
    
    # Calculate standard deviation around median
    std_median_leg = np.std(resid_data_leg.flatten() -\
                            np.median(resid_data_leg.flatten()))
    std_median_sin = np.std(resid_data_sin.flatten() -\
                            np.median(resid_data_sin.flatten()))
    
    print(std_median_leg)
    print(std_median_sin)
    
    # Do this, but just for central arcsecond
    # First determine Sgr A* position from align rms, if it exists
    if not os.path.exists(align_root + '.sgra'):
        print(f"Align RMS for {epoch_name} in {filt_name} not completed")
        return
    
    sgra_table = Table.read(
        align_root + '.sgra',
        format='ascii.commented_header',
        header_start=2,
    )
    
    # Construct arrays of x and y coordinates
    (image_y_len, image_x_len) = image_data_orig.shape
    
    image_x_coords = np.arange(0, image_x_len,),
    image_y_coords = np.arange(0, image_y_len,),
    
    image_x_array_coords, image_y_array_coords, = np.meshgrid(
        image_x_coords,
        image_y_coords,
    )
    
    # Determine x and y difference from Sgr A* position
    sgra_x_diff = image_x_array_coords - sgra_table['X'][0]
    sgra_y_diff = image_y_array_coords - sgra_table['Y'][0]
    
    # Make a cut where distance is less than cut radius
    cut_pixel_radius = 1 / plate_scale
    
    cent_arcsec_cut = np.where(
        np.hypot(
            sgra_x_diff, sgra_y_diff,
        ) < cut_pixel_radius
    )
    
    # Implement cut on image and resid data
    image_data_cut = image_data_orig[cent_arcsec_cut]
    resid_data_leg_cut = resid_data_leg_orig[cent_arcsec_cut]
    resid_data_sin_cut = resid_data_sin_orig[cent_arcsec_cut]
    
    # Save out data
    out_table = Table(
        [
            [np.nanmedian(np.sqrt(resid_data_leg_cut.flatten() ** 2))],
            [np.nanmedian(np.sqrt(resid_data_sin_cut.flatten() ** 2))],
            [np.nanmedian(np.sqrt(resid_data_leg_cut.flatten()**2) / image_data_cut.flatten())],
            [np.nanmedian(np.sqrt(resid_data_sin_cut.flatten()**2) / image_data_cut.flatten())],
        ],
        names=(
            'resid_abs_leg',
            'resid_abs_sin',
            'resid_abs_feu_leg',
            'resid_abs_feu_sin',
        ),
    )
    
    out_table.write(
        plot_out_dir + 'resids_centarcsec.txt',
        format='ascii.fixed_width_two_line',
        overwrite=True,
    )
    out_table.write(
        plot_out_dir + 'resids_centarcsec.h5',
        format='hdf5', path='data', serialize_meta=True,
        overwrite=True,
    )
    
    # Draw histogram
    plt.style.use(['ticks_outtie'])
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    hist_bins = np.linspace(0, 1000, num=51)
    
    ax.hist(np.sqrt(resid_data_leg_cut.flatten() ** 2), bins=hist_bins,
            histtype='step',
            color='C0', label='Legacy')
    
    ax.hist(np.sqrt(resid_data_sin_cut.flatten() ** 2), bins=hist_bins,
            histtype='step',
            color='C1', label='Single-PSF')
    
    ax.axvline(np.nanmedian(np.sqrt(resid_data_leg_cut.flatten()**2)),
               color='C0', ls='--')
    ax.axvline(np.nanmedian(np.sqrt(resid_data_sin_cut.flatten()**2)),
               color='C1', ls='--')
    
    print(np.nanmedian(np.sqrt(resid_data_leg_cut.flatten()**2)))
    print(np.nanmedian(np.sqrt(resid_data_sin_cut.flatten()**2)))
    
    ax.set_xlabel(r'$\sqrt{(data - model)^2}$')
    ax.set_ylabel('Number of pixels')
    
    ax.set_xlim([0, 1000])
    
    ax.legend()
    
    x_majorLocator = MultipleLocator(200)
    x_minorLocator = MultipleLocator(50)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)
    
    y_majorLocator = MultipleLocator(2e4)
    y_minorLocator = MultipleLocator(5e3)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()
    
    fig.savefig(plot_out_dir + 'resid_hist_centarcsec.pdf')
    fig.savefig(plot_out_dir + 'resid_hist_centarcsec.png', dpi=200)
    
    plt.close(fig)
    
    # Draw histogram
    plt.style.use(['ticks_outtie'])
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    
    hist_bins = np.linspace(0, 5, num=51)
    # hist_bins = np.linspace(0, 10, num=80)
    # hist_bins = np.logspace(-5, 2, num=100)
    
    
    ax.hist(np.sqrt(resid_data_leg_cut.flatten()**2) / image_data_cut.flatten(),
            bins=hist_bins,
            histtype='step',
            color='C0', label='Legacy')
    
    ax.hist(np.sqrt(resid_data_sin_cut.flatten()**2) / image_data_cut.flatten(),
            bins=hist_bins,
            histtype='step',
            color='C1', label='Single-PSF')
    
    print(np.nanmedian(np.sqrt(resid_data_leg_cut.flatten()**2) / image_data_cut.flatten()))
    print(np.nanmedian(np.sqrt(resid_data_sin_cut.flatten()**2) / image_data_cut.flatten()))
    
    ax.axvline(np.nanmedian(np.sqrt(resid_data_leg_cut.flatten()**2) / image_data_cut.flatten()),
               color='C0', ls='--')
    ax.axvline(np.nanmedian(np.sqrt(resid_data_sin_cut.flatten()**2) / image_data_cut.flatten()),
               color='C1', ls='--')
    
    ax.set_xlabel(r'$\sqrt{(data-model)^2}$ / $data$')
    ax.set_ylabel('Number of pixels')
    
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    
    ax.set_xlim([0, 5])
    
    # ax.legend(loc='upper left')
    ax.legend()
    
    x_majorLocator = MultipleLocator(1)
    x_minorLocator = MultipleLocator(0.2)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)
    
    y_majorLocator = MultipleLocator(5e4)
    y_minorLocator = MultipleLocator(1e4)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()
    
    fig.savefig(plot_out_dir + 'resid_feu_hist_centarcsec.pdf')
    fig.savefig(plot_out_dir + 'resid_feu_hist_centarcsec.png', dpi=200)
    
    plt.close(fig)


# Read in epochs table
epochs_table = Table.read('epochs_table.h5', format='hdf5', path='data')

# epochs_table = epochs_table[np.where(epochs_table['nights_combo'] == 'single_night')]

# resid_hist_plotter('20160503nirc2', dr_path = dr_path,
#                    filt_name = 'kp', stf_corr = stf_corr,
#                    legacy_version_str = legacy_version_str,
#                    single_version_str = single_version_str)


# Run analysis code on all epochs
for epochs_row in tqdm(epochs_table):
    cur_epoch = epochs_row['epoch']
    cur_filt = epochs_row['filt']

    resid_hist_plotter(
        cur_epoch, dr_path = dr_path,
        filt_name = cur_filt, stf_corr = stf_corr,
        legacy_version_str = legacy_version_str,
        single_version_str = single_version_str,
    )

    # break
