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

import matplotlib as mpl
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

faint_mag = {'kp': 17,
             'h': 19,
             'lp': 15,
            }

# Read in epochs table
epochs_table = Table.read('epochs_table.h5', format='hdf5', path='data')

epochs_table = epochs_table[np.where(epochs_table['nights_combo'] == 'single_night')]
epochs_table = epochs_table[np.where(epochs_table['filt'] == 'kp')]

cur_wd = os.getcwd()

plot_out_dir = cur_wd + '/align_rms_depth_comparison_plots/'
os.makedirs(plot_out_dir, exist_ok=True)

num_epochs = len(epochs_table)

# Empty arrays for detection numbers
num_stf_legacy_detections = np.empty(num_epochs)
num_stf_singPSF_detections = np.empty(num_epochs)

num_stf_legacy_detections_faint = np.empty(num_epochs)
num_stf_singPSF_detections_faint = np.empty(num_epochs)

num_legonly_detections = np.empty(num_epochs)
num_singonly_detections = np.empty(num_epochs)

num_legonly_faint_detections = np.empty(num_epochs)
num_singonly_faint_detections = np.empty(num_epochs)

construct_table = True

# Run analysis code on all epochs
if construct_table:
    for epoch_index in tqdm(range(num_epochs)):
        epochs_row = epochs_table[epoch_index]
    
        epoch_name = epochs_row['epoch']
        filt_name = epochs_row['filt']
    
    
        epoch_analysis_location = '{0}/{1}_{2}/'.format(cur_wd, epoch_name, filt_name)
        starlist_align_location = epoch_analysis_location + 'starlists_align_rms/align/'
        align_root = starlist_align_location + 'align_d_rms_abs'

        # plot_out_dir = epoch_analysis_location + 'align_rms_mag_comparison_plots/'
        # os.makedirs(plot_out_dir, exist_ok=True)

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
    
        # Determine number of detections
        num_stf_legacy_detections[epoch_index] = len(stf_legacy_lis_table)
        num_stf_singPSF_detections[epoch_index] = len(stf_singPSF_lis_table)
    
        # Determine number of faint detections
        faint_legacy_filter = np.where(stf_legacy_lis_table['m'] >= faint_mag[filt_name])
        faint_singPSF_filter = np.where(stf_singPSF_lis_table['m'] >= faint_mag[filt_name])
    
        num_stf_legacy_detections_faint[epoch_index] =\
            len(stf_legacy_lis_table[faint_legacy_filter])
        num_stf_singPSF_detections_faint[epoch_index] =\
            len(stf_singPSF_lis_table[faint_singPSF_filter])
    
        # Determine number of unique (one mode) detections
        num_legonly_detections[epoch_index] =\
            len(legonly_detections_mag_table)
        num_singonly_detections[epoch_index] =\
            len(singonly_detections_mag_table)
        
        legonly_faint_filter = np.where(legonly_detections_mag_table[legacy_version_str + '_mag'] >= faint_mag[filt_name])
        singonly_faint_filter = np.where(singonly_detections_mag_table[single_version_str + '_mag'] >= faint_mag[filt_name])
    
        num_legonly_faint_detections[epoch_index] =\
            len(legonly_detections_mag_table[legonly_faint_filter])
        num_singonly_faint_detections[epoch_index] =\
            len(singonly_detections_mag_table[singonly_faint_filter])

    epochs_table.add_columns(
        [num_stf_legacy_detections, num_stf_singPSF_detections,
         num_stf_legacy_detections_faint, num_stf_singPSF_detections_faint,
         num_legonly_detections, num_singonly_detections,
         num_legonly_faint_detections, num_singonly_faint_detections],
        names=['num_stf_legacy_detections', 'num_stf_singPSF_detections',
               'num_stf_legacy_detections_faint',
               'num_stf_singPSF_detections_faint',
               'num_legonly_detections', 'num_singonly_detections',
               'num_legonly_faint_detections', 'num_singonly_faint_detections',
              ],
    )

    epochs_table.write(f'{plot_out_dir}/epochs_table.txt',
        format='ascii.fixed_width_two_line', overwrite=True)
    
    epochs_table.write(f'{plot_out_dir}/epochs_table.h5', format='hdf5',
                       path='data', serialize_meta=True,
                       overwrite=True)
else:
    epochs_table = Table.read(f'{plot_out_dir}/epochs_table.h5',
                              format='hdf5', path='data')

# Draw plots


# Number of detections vs. number of frames
plt.style.use(['tex_paper', 'ticks_outtie'])
fig, ax = plt.subplots(figsize=(4,4))

ax.plot(epochs_table['num_frames'],
        epochs_table['num_stf_legacy_detections'],
        'o', alpha=0.6, color='C0', label='Legacy')
ax.plot(epochs_table['num_frames'],
        epochs_table['num_stf_singPSF_detections'],
        'o', alpha=0.6, color='C1', label='Single-PSF')

ax.set_xlabel('Number of frames in combo')
ax.set_ylabel('Number of stars detected')

ax.legend(loc='upper left')

x_majorLocator = MultipleLocator(50)
x_minorLocator = MultipleLocator(10)
ax.xaxis.set_major_locator(x_majorLocator)
ax.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(250)
y_minorLocator = MultipleLocator(50)
ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_minor_locator(y_minorLocator)

ax.set_xlim([0, 200])
ax.set_ylim([0, 2250])

# Save out and close figure
fig.tight_layout()

fig.savefig(plot_out_dir + 'num_frames_dets_tex.pdf')
fig.savefig(plot_out_dir + 'num_frames_dets_tex.png', dpi=200)

plt.close(fig)

# Number of faint detections vs. number of frames
plt.style.use(['tex_paper', 'ticks_outtie'])
fig, ax = plt.subplots(figsize=(4,4))

ax.plot(epochs_table['num_frames'],
        epochs_table['num_stf_legacy_detections_faint'],
        'o', alpha=0.6, color='C0', label='Legacy')
ax.plot(epochs_table['num_frames'],
        epochs_table['num_stf_singPSF_detections_faint'],
        'o', alpha=0.6, color='C1', label='Single-PSF')

ax.set_xlabel('Number of frames in combo')
ax.set_ylabel(f"No. of faint stars ({filt_mag_label_strs['kp']} $\\geq$ {faint_mag['kp']}) detected")

ax.legend(loc='upper left')

x_majorLocator = MultipleLocator(50)
x_minorLocator = MultipleLocator(10)
ax.xaxis.set_major_locator(x_majorLocator)
ax.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(250)
y_minorLocator = MultipleLocator(50)
ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_minor_locator(y_minorLocator)

ax.set_xlim([0, 200])
ax.set_ylim([0, 2250])

# Save out and close figure
fig.tight_layout()

fig.savefig(plot_out_dir + 'num_frames_dets_faint_tex.pdf')
fig.savefig(plot_out_dir + 'num_frames_dets_faint_tex.png', dpi=200)

plt.close(fig)

# Number of detections vs. median Strehl
plt.style.use(['tex_paper', 'ticks_outtie'])
fig, ax = plt.subplots(figsize=(4,4))

ax.plot(epochs_table['med_strehls'],
        epochs_table['num_stf_legacy_detections'],
        'o', alpha=0.6, color='C0', label='Legacy')
ax.plot(epochs_table['med_strehls'],
        epochs_table['num_stf_singPSF_detections'],
        'o', alpha=0.6, color='C1', label='Single-PSF')

ax.set_xlabel('Median Strehl ratio of frames in combo')
ax.set_ylabel('Number of stars detected')

ax.legend(loc='upper left')

x_majorLocator = MultipleLocator(0.1)
x_minorLocator = MultipleLocator(0.02)
ax.xaxis.set_major_locator(x_majorLocator)
ax.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(250)
y_minorLocator = MultipleLocator(50)
ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_minor_locator(y_minorLocator)

ax.set_xlim([0, 0.5])
ax.set_ylim([0, 2250])

# Save out and close figure
fig.tight_layout()

fig.savefig(plot_out_dir + 'med_strehls_dets_tex.pdf')
fig.savefig(plot_out_dir + 'med_strehls_dets_tex.png', dpi=200)

plt.close(fig)

# Number of faint detections vs. median Strehl
plt.style.use(['tex_paper', 'ticks_outtie'])
fig, ax = plt.subplots(figsize=(4,4))

ax.plot(epochs_table['med_strehls'],
        epochs_table['num_stf_legacy_detections_faint'],
        'o', alpha=0.6, color='C0', label='Legacy')
ax.plot(epochs_table['med_strehls'],
        epochs_table['num_stf_singPSF_detections_faint'],
        'o', alpha=0.6, color='C1', label='Single-PSF')

ax.set_xlabel('Median Strehl ratio of frames in combo')
ax.set_ylabel(f"No. of faint stars ({filt_mag_label_strs['kp']} $\\geq$ {faint_mag['kp']}) detected")

ax.legend(loc='upper left')

x_majorLocator = MultipleLocator(0.1)
x_minorLocator = MultipleLocator(0.02)
ax.xaxis.set_major_locator(x_majorLocator)
ax.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(250)
y_minorLocator = MultipleLocator(50)
ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_minor_locator(y_minorLocator)

ax.set_xlim([0, 0.5])
ax.set_ylim([0, 2250])

# Save out and close figure
fig.tight_layout()

fig.savefig(plot_out_dir + 'med_strehls_dets_faint_tex.pdf')
fig.savefig(plot_out_dir + 'med_strehls_dets_faint_tex.png', dpi=200)

plt.close(fig)

# Number of detections vs. median FWHM
plt.style.use(['tex_paper', 'ticks_outtie'])
fig, ax = plt.subplots(figsize=(4,4))

ax.plot(epochs_table['med_fwhms'],
        epochs_table['num_stf_legacy_detections'],
        'o', alpha=0.6, color='C0', label='Legacy')
ax.plot(epochs_table['med_fwhms'],
        epochs_table['num_stf_singPSF_detections'],
        'o', alpha=0.6, color='C1', label='Single-PSF')

ax.set_xlabel('Median FWHM (mas) of frames in combo')
ax.set_ylabel('Number of stars detected')

ax.legend(loc='upper right')

x_majorLocator = MultipleLocator(20)
x_minorLocator = MultipleLocator(5)
ax.xaxis.set_major_locator(x_majorLocator)
ax.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(250)
y_minorLocator = MultipleLocator(50)
ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_minor_locator(y_minorLocator)

ax.set_xlim([40, 120])
ax.set_ylim([0, 2250])

# Save out and close figure
fig.tight_layout()

fig.savefig(plot_out_dir + 'med_fwhms_dets_tex.pdf')
fig.savefig(plot_out_dir + 'med_fwhms_dets_tex.png', dpi=200)

plt.close(fig)

# Number of faint detections vs. median FWHM
plt.style.use(['tex_paper', 'ticks_outtie'])
fig, ax = plt.subplots(figsize=(4,4))

ax.plot(epochs_table['med_fwhms'],
        epochs_table['num_stf_legacy_detections_faint'],
        'o', alpha=0.6, color='C0', label='Legacy')
ax.plot(epochs_table['med_fwhms'],
        epochs_table['num_stf_singPSF_detections_faint'],
        'o', alpha=0.6, color='C1', label='Single-PSF')

ax.set_xlabel('Median FWHM (mas) of frames in combo')
ax.set_ylabel(f"No. of faint stars ({filt_mag_label_strs['kp']} $\\geq$ {faint_mag['kp']}) detected")

ax.legend(loc='upper right')

x_majorLocator = MultipleLocator(20)
x_minorLocator = MultipleLocator(5)
ax.xaxis.set_major_locator(x_majorLocator)
ax.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(250)
y_minorLocator = MultipleLocator(50)
ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_minor_locator(y_minorLocator)

ax.set_xlim([40, 120])
ax.set_ylim([0, 2250])

# Save out and close figure
fig.tight_layout()

fig.savefig(plot_out_dir + 'med_fwhms_dets_faint_tex.pdf')
fig.savefig(plot_out_dir + 'med_fwhms_dets_faint_tex.png', dpi=200)

plt.close(fig)

# Number of unique detections vs. median Strehl
plt.style.use(['tex_paper', 'ticks_outtie'])
fig, ax = plt.subplots(figsize=(4,4))

ax.plot(epochs_table['med_strehls'],
        epochs_table['num_legonly_detections'],
        'o', alpha=0.6, color='C0', label='Legacy Only')
ax.plot(epochs_table['med_strehls'],
        epochs_table['num_singonly_detections'],
        'o', alpha=0.6, color='C1', label='Single-PSF Only')

ax.set_xlabel('Median Strehl ratio of frames in combo')
ax.set_ylabel(f"Number of stars detected")

ax.legend(loc='upper left')

x_majorLocator = MultipleLocator(0.1)
x_minorLocator = MultipleLocator(0.02)
ax.xaxis.set_major_locator(x_majorLocator)
ax.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(250)
y_minorLocator = MultipleLocator(50)
ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_minor_locator(y_minorLocator)

ax.set_xlim([0, 0.5])
ax.set_ylim([0, 2250])

# Save out and close figure
fig.tight_layout()

fig.savefig(plot_out_dir + 'med_strehls_dets_unique_tex.pdf')
fig.savefig(plot_out_dir + 'med_strehls_dets_unique_tex.png', dpi=200)

plt.close(fig)


# Number of unique faint detections vs. median Strehl
plt.style.use(['tex_paper', 'ticks_outtie'])
fig, ax = plt.subplots(figsize=(4,4))

ax.plot(epochs_table['med_strehls'],
        epochs_table['num_legonly_faint_detections'],
        'o', alpha=0.6, color='C0', label='Legacy Only')
ax.plot(epochs_table['med_strehls'],
        epochs_table['num_singonly_faint_detections'],
        'o', alpha=0.6, color='C1', label='Single-PSF Only')

ax.set_xlabel('Median Strehl ratio of frames in combo')
ax.set_ylabel(f"No. of faint stars ({filt_mag_label_strs['kp']} $\\geq$ {faint_mag['kp']}) detected")

ax.legend(loc='upper left')

x_majorLocator = MultipleLocator(0.1)
x_minorLocator = MultipleLocator(0.02)
ax.xaxis.set_major_locator(x_majorLocator)
ax.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(250)
y_minorLocator = MultipleLocator(50)
ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_minor_locator(y_minorLocator)

ax.set_xlim([0, 0.5])
ax.set_ylim([0, 2250])

# Save out and close figure
fig.tight_layout()

fig.savefig(plot_out_dir + 'med_strehls_dets_unique_faint_tex.pdf')
fig.savefig(plot_out_dir + 'med_strehls_dets_unique_faint_tex.png', dpi=200)

plt.close(fig)


# Comparison of detections in different modes, colored by quality
plt.style.use(['tex_paper', 'ticks_innie'])
fig, ax = plt.subplots(figsize=(5, 4))

ax.plot([0, 2250], [0, 2250], '--', lw=0.5, color='k', alpha=0.8)

scatter_norm = mpl.colors.Normalize(vmin=0.1, vmax=0.5)
scatter_cmap = 'plasma'

ax.scatter(epochs_table['num_stf_legacy_detections'],
           epochs_table['num_stf_singPSF_detections'],
           c=epochs_table['med_strehls'],
           norm=scatter_norm,
           cmap=scatter_cmap, zorder=2.5,
          )

fig.colorbar(mpl.cm.ScalarMappable(norm=scatter_norm, cmap=scatter_cmap),
             ax=ax, label='Median Strehl ratio of frames in combo')

ax.set_xlabel(f"No. stars detected, Legacy mode")
ax.set_ylabel(f"No. stars detected, Single-PSF mode")

ax.set_xlim([0, 2250])
ax.set_ylim([0, 2250])

ax.set_aspect('equal', 'box')

ax.grid(b=True, which='major', ls='--', lw=0.5)

x_majorLocator = MultipleLocator(500)
x_minorLocator = MultipleLocator(100)
ax.xaxis.set_major_locator(x_majorLocator)
ax.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(500)
y_minorLocator = MultipleLocator(100)
ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_minor_locator(y_minorLocator)


# Save out and close figure
fig.tight_layout()

fig.savefig(plot_out_dir + 'det_comparison_med_strehls.pdf')
fig.savefig(plot_out_dir + 'det_comparison_med_strehls.png', dpi=200)

plt.close(fig)

# No. stars detected leg, sing-psf, median FWHM 

plt.style.use(['tex_paper', 'ticks_innie'])
fig, ax = plt.subplots(figsize=(5, 4))

ax.plot([0, 2250], [0, 2250], '--', lw=0.5, color='k', alpha=0.8)

scatter_norm = mpl.colors.Normalize(vmin=40, vmax=120)
scatter_norm = mpl.colors.LogNorm(vmin=40, vmax=120)
scatter_cmap = 'plasma_r'

ax.scatter(epochs_table['num_stf_legacy_detections'],
           epochs_table['num_stf_singPSF_detections'],
           c=epochs_table['med_fwhms'],
           norm=scatter_norm,
           cmap=scatter_cmap, zorder=2.5,
          )

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=scatter_norm, cmap=scatter_cmap),
                    ax=ax, label='Median FWHM (mas) of frames in combo',
                    ticks=[40, 60, 80, 100, 120])
cbar.ax.set_yticklabels(['40', '60', '80', '100', '120'])

ax.set_xlabel(f"No. stars detected, Legacy mode")
ax.set_ylabel(f"No. stars detected, Single-PSF mode")

ax.set_xlim([0, 2250])
ax.set_ylim([0, 2250])

ax.set_aspect('equal', 'box')

ax.grid(b=True, which='major', ls='--', lw=0.5)

x_majorLocator = MultipleLocator(500)
x_minorLocator = MultipleLocator(100)
ax.xaxis.set_major_locator(x_majorLocator)
ax.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(500)
y_minorLocator = MultipleLocator(100)
ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_minor_locator(y_minorLocator)


# Save out and close figure
fig.tight_layout()

fig.savefig(plot_out_dir + 'det_comparison_med_fwhms.pdf')
fig.savefig(plot_out_dir + 'det_comparison_med_fwhms.png', dpi=200)

plt.close(fig)

# No. stars detected leg, sing-psf, number of frames

plt.style.use(['tex_paper', 'ticks_innie'])
fig, ax = plt.subplots(figsize=(5, 4))

ax.plot([0, 2250], [0, 2250], '--', lw=0.5, color='k', alpha=0.8)

scatter_norm = mpl.colors.Normalize(vmin=0, vmax=200)
# scatter_norm = mpl.colors.LogNorm(vmin=4, vmax=200)
scatter_cmap = 'plasma'

ax.scatter(epochs_table['num_stf_legacy_detections'],
           epochs_table['num_stf_singPSF_detections'],
           c=epochs_table['num_frames'],
           norm=scatter_norm,
           cmap=scatter_cmap, zorder=2.5,
          )

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=scatter_norm, cmap=scatter_cmap),
                    ax=ax, label='Number of frames in combo',
                    # ticks=[4, 10, 20, 50, 100, 110, 120, 150, 200],
                   )
# cbar.ax.set_yticklabels(['4', '10', '20', '50', '100',
#                          '110', '120', '150', '200'])

ax.set_xlabel(f"No. stars detected, Legacy mode")
ax.set_ylabel(f"No. stars detected, Single-PSF mode")

ax.set_xlim([0, 2250])
ax.set_ylim([0, 2250])

ax.set_aspect('equal', 'box')

ax.grid(b=True, which='major', ls='--', lw=0.5)

x_majorLocator = MultipleLocator(500)
x_minorLocator = MultipleLocator(100)
ax.xaxis.set_major_locator(x_majorLocator)
ax.xaxis.set_minor_locator(x_minorLocator)

y_majorLocator = MultipleLocator(500)
y_minorLocator = MultipleLocator(100)
ax.yaxis.set_major_locator(y_majorLocator)
ax.yaxis.set_minor_locator(y_minorLocator)


# Save out and close figure
fig.tight_layout()

fig.savefig(plot_out_dir + 'det_comparison_num_frames.pdf')
fig.savefig(plot_out_dir + 'det_comparison_num_frames.png', dpi=200)

plt.close(fig)

