#!/usr/bin/env python

import os
import copy
import numpy as np
from scipy import stats
from astropy.table import Table
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def construct_combined_hist_table(
        epoch_list, epoch_jyears, epoch_tables,
        filt, out_name,
        cut_mag_lo=14, cut_mag_hi=17,
    ):
    
    # Use first epoch to construct integrated table
    integrated_table = copy.deepcopy(epoch_tables[epoch_list[0]])
    
    # Go through rest of epochs, and add to integrated table
    for cur_epoch in epoch_list[1:]:
        cur_epoch_table = epoch_tables[cur_epoch]
        
        integrated_table['num_dets_common'] =\
            integrated_table['num_dets_common'] + cur_epoch_table['num_dets_common']
        
        integrated_table['num_dets_legonly'] =\
            integrated_table['num_dets_legonly'] + cur_epoch_table['num_dets_legonly']
        
        integrated_table['num_dets_sinonly'] =\
            integrated_table['num_dets_sinonly'] + cur_epoch_table['num_dets_sinonly']
    
    # Write out integrated table
    integrated_table.write(
        out_name + '_int_table.txt',
        format='ascii.fixed_width_two_line',
        overwrite=True,
    )
    integrated_table.write(
        out_name + '_int_table.h5',
        format='hdf5', path='data', serialize_meta=True,
        overwrite=True,
    )
    
    # Plot integrated histogram
    plt.style.use(['ticks_outtie'])
    
    plot_bin_edges = np.append(
        integrated_table['mag_bin_lo'],
        integrated_table['mag_bin_hi'][-1]
    )
    
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    ax.stairs(
        integrated_table['num_dets_common'],
        edges=plot_bin_edges,
        fill=False,
        color='k', label='Detections in both modes',
    )
    
    ax.stairs(
        integrated_table['num_dets_legonly'],
        edges=plot_bin_edges,
        fill=False, hatch='///',
        color='C0', label='Legacy only detections',
    )
    
    ax.stairs(
        integrated_table['num_dets_sinonly'],
        edges=plot_bin_edges,
        fill=False, hatch='\\\\\\',
        color='C1', label='Single-PSF only detections',
    )
    
    ax.set_xlabel(filt + ' Mag')
    ax.set_ylabel('Integrated Detections')
    
    ax.legend(loc='upper left')
    
    ax.set_xlim([
        np.min(integrated_table['mag_bin_lo']),
        np.max(integrated_table['mag_bin_hi']),
    ])
    
    x_majorLocator = MultipleLocator(2)
    x_minorLocator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(50)
    y_minorLocator = MultipleLocator(10)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()

    fig.savefig(out_name + '_int_hist.pdf')
    fig.savefig(out_name + '_int_hist.png', dpi=200)

    plt.close(fig)
    
    # Epoch by epoch, within cut mags
    num_epochs = len(epoch_list)
    
    cut_mags_dets_common = np.empty(num_epochs)
    cut_mags_dets_legonly = np.empty(num_epochs)
    cut_mags_dets_sinonly = np.empty(num_epochs)
    
    for (epoch_index, epoch_name) in enumerate(epoch_list):
        # Filter to just the cut mags
        cur_epoch_table = epoch_tables[epoch_name]
        
        cut_filt = np.where(
            np.logical_and(
                cur_epoch_table['mag_bin_lo'] >= cut_mag_lo,
                cur_epoch_table['mag_bin_hi'] <= cut_mag_hi,
            )
        )
        
        cut_mags_epoch_table = cur_epoch_table[cut_filt]
        
        # Store out sum of all detections within the cut mags
        cut_mags_dets_common[epoch_index] = np.sum(
            cut_mags_epoch_table['num_dets_common']
        )
        cut_mags_dets_legonly[epoch_index] = np.sum(
            cut_mags_epoch_table['num_dets_legonly']
        )
        cut_mags_dets_sinonly[epoch_index] = np.sum(
            cut_mags_epoch_table['num_dets_sinonly']
        )
    
    # Construct and write out table for detections
    cut_mags_table = Table(
        [
            epoch_list, epoch_jyears,
            cut_mags_dets_common, cut_mags_dets_legonly, cut_mags_dets_sinonly,
        ],
        names=(
            'epoch',
            'jyear',
            'num_dets_common',
            'num_dets_legonly',
            'num_dets_sinonly',
        ),
    )
    
    cut_mags_table.write(
        out_name + f'_cut_mags_{cut_mag_lo}_{cut_mag_hi}.txt',
        format='ascii.fixed_width_two_line',
        overwrite=True,
    )
    cut_mags_table.write(
        out_name + f'_cut_mags_{cut_mag_lo}_{cut_mag_hi}.h5',
        format='hdf5', path='data', serialize_meta=True,
        overwrite=True,
    )
    
    # Draw plot showing detections among cut mags
    plt.style.use(['ticks_outtie'])
    
    fig, ax = plt.subplots(figsize=(6,3), frameon=False)
    ax.plot(
        epoch_jyears,
        cut_mags_dets_common,
        'o',
        color='k', label='Detections in both modes',
    )
    
    ax.plot(
        epoch_jyears,
        cut_mags_dets_legonly,
        'o',
        color='C0', label='Legacy only detections',
    )
    
    ax.plot(
        epoch_jyears,
        cut_mags_dets_sinonly,
        'o',
        color='C1', label='Single-PSF only detections',
    )
    
    ax.set_xlabel('Observation Date')
    ax.set_ylabel(f'Detection num: {cut_mag_lo} <= {filt} <= {cut_mag_hi}')
    
    ax.legend(
        loc='upper left',
        ncol=3, fontsize='x-small',
        bbox_to_anchor=(0, -0.27 1, 0.05), mode='expand',
    )
    
    ax.set_xlim([2006, 2024])
    
    x_majorLocator = MultipleLocator(4)
    x_minorLocator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_majorLocator)
    ax.xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(10)
    y_minorLocator = MultipleLocator(2)
    ax.yaxis.set_major_locator(y_majorLocator)
    ax.yaxis.set_minor_locator(y_minorLocator)
    
    fig.tight_layout()

    fig.savefig(out_name + f'_cut_mags_{cut_mag_lo}_{cut_mag_hi}.pdf')
    fig.savefig(out_name + f'_cut_mags_{cut_mag_lo}_{cut_mag_hi}.png', dpi=200)

    plt.close(fig)

# Read list of epochs for key project
epochs_table = Table.read(
    './key_project_epochs.txt', format='ascii.commented_header',
)

print(epochs_table)

# Read in combo epochs quality table to get JYears
dr_path = '/g/ghez/data/dr/dr1/'

combo_quality_table_kp = Table.read(
    dr_path + 'data_quality/combo_epochs_quality/combo_epochs_quality_table_kp.h5',
    format='hdf5', path='data',
)
combo_quality_table_kp.add_index('epoch')

combo_quality_table_h = Table.read(
    dr_path + 'data_quality/combo_epochs_quality/combo_epochs_quality_table_h.h5',
    format='hdf5', path='data',
)
combo_quality_table_h.add_index('epoch')


mag_num_hist_tables = {}
mag_num_hist_centarcsec_tables = {}

kp_epochs_complete = []
h_epochs_complete = []

kp_jyear = []
h_jyear = []

for epoch_row in tqdm(epochs_table):
    epoch_combo_name = epoch_row['epoch'] + '_' + epoch_row['filter']
    
    # Check if histogram file exists
    hist_file = f'../{epoch_combo_name}/align_rms_pos_num_comparison_plots/mag_num_hist.h5'
    
    if not os.path.exists(hist_file):
        print(f"align rms comparison on epoch {epoch_combo_name} not yet run")
        continue
    else:
        if epoch_row['filter'] == 'kp':
            kp_epochs_complete.append(epoch_combo_name)
            kp_jyear.append(
                (combo_quality_table_kp.loc[epoch_row['epoch']])['JYear']
            )
        elif epoch_row['filter'] == 'h':
            h_epochs_complete.append(epoch_combo_name)
            h_jyear.append(
                (combo_quality_table_h.loc[epoch_row['epoch']])['JYear']
            )
    
    # Read in tables, and store
    cur_mag_num_hist_table = Table.read(
        hist_file, format='hdf5', path='data',
    )
    
    hist_file = f'../{epoch_combo_name}/align_rms_pos_num_comparison_plots/mag_num_hist_centarcsec.h5'
    
    cur_mag_num_hist_centarcsec_table = Table.read(
        hist_file, format='hdf5', path='data',
    )
    
    mag_num_hist_tables[epoch_combo_name] = cur_mag_num_hist_table
    mag_num_hist_centarcsec_tables[epoch_combo_name] = cur_mag_num_hist_centarcsec_table

# Draw plots and make tables
construct_combined_hist_table(
    kp_epochs_complete, kp_jyear, mag_num_hist_centarcsec_tables,
    'kp', 'kp_centarcsec',
    cut_mag_lo=14, cut_mag_hi=17,
)

construct_combined_hist_table(
    kp_epochs_complete, kp_jyear, mag_num_hist_centarcsec_tables,
    'kp', 'kp_centarcsec',
    cut_mag_lo=14, cut_mag_hi=16,
)

construct_combined_hist_table(
    h_epochs_complete, h_jyear, mag_num_hist_centarcsec_tables,
    'h', 'h_centarcsec',
    cut_mag_lo=16, cut_mag_hi=19,
)

construct_combined_hist_table(
    h_epochs_complete, h_jyear, mag_num_hist_centarcsec_tables,
    'h', 'h_centarcsec',
    cut_mag_lo=16, cut_mag_hi=18,
)