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
        epoch_list, epoch_jyears,
        epoch_tables, epoch_tables_ca,
        filt, out_name,
    ):
    
    print('---')
    print(out_name)
    
    # Epoch by epoch
    num_epochs = len(epoch_list)
    
    resids_abs_leg = np.empty(num_epochs)
    resids_abs_sin = np.empty(num_epochs)
    resids_abs_feu_leg = np.empty(num_epochs)
    resids_abs_feu_sin = np.empty(num_epochs)
    
    for (epoch_index, epoch_name) in enumerate(epoch_list):
        cur_epoch_table = epoch_tables[epoch_name]
        
        # Store out the resids
        resids_abs_leg[epoch_index] = cur_epoch_table['resid_abs_leg'][0]
        resids_abs_sin[epoch_index] = cur_epoch_table['resid_abs_sin'][0]
        resids_abs_feu_leg[epoch_index] = cur_epoch_table['resid_abs_feu_leg'][0]
        resids_abs_feu_sin[epoch_index] = cur_epoch_table['resid_abs_feu_sin'][0]
        
        if resids_abs_sin[epoch_index] > resids_abs_leg[epoch_index]:
            print(f"Entire image: {epoch_name}")
    
    hl_filt = np.where(np.char.endswith(epoch_list, '_h'))
    
    print(f'Mean abs, legacy = {np.mean(resids_abs_leg):.5f}')
    print(f'Mean abs, single-PSF = {np.mean(resids_abs_sin):.5f}')
    
    print(f'Mean abs FEU, legacy = {np.mean(resids_abs_feu_leg):.5f}')
    print(f'Mean abs FEU, single-PSF = {np.mean(resids_abs_feu_sin):.5f}')
    
    resids_abs_leg_ca = np.empty(num_epochs)
    resids_abs_sin_ca = np.empty(num_epochs)
    resids_abs_feu_leg_ca = np.empty(num_epochs)
    resids_abs_feu_sin_ca = np.empty(num_epochs)
    
    for (epoch_index, epoch_name) in enumerate(epoch_list):
        cur_epoch_table_ca = epoch_tables_ca[epoch_name]
        
        # Store out the resids
        resids_abs_leg_ca[epoch_index] = cur_epoch_table_ca['resid_abs_leg'][0]
        resids_abs_sin_ca[epoch_index] = cur_epoch_table_ca['resid_abs_sin'][0]
        resids_abs_feu_leg_ca[epoch_index] = cur_epoch_table_ca['resid_abs_feu_leg'][0]
        resids_abs_feu_sin_ca[epoch_index] = cur_epoch_table_ca['resid_abs_feu_sin'][0]
        
        if resids_abs_sin_ca[epoch_index] > resids_abs_leg_ca[epoch_index]:
            print(f"Central arcsec: {epoch_name}")    
    
    # Construct and write out table for detections
    resids_table = Table(
        [
            epoch_list, epoch_jyears,
            resids_abs_leg,
            resids_abs_sin,
            resids_abs_feu_leg,
            resids_abs_feu_sin,
        ],
        names=(
            'epoch',
            'jyear',
            'resid_abs_leg',
            'resid_abs_sin',
            'resid_abs_feu_leg',
            'resid_abs_feu_sin',
        ),
    )
    
    resids_table.write(
        out_name + '_resids.txt',
        format='ascii.fixed_width_two_line',
        overwrite=True,
    )
    resids_table.write(
        out_name + f'_resids.h5',
        format='hdf5', path='data', serialize_meta=True,
        overwrite=True,
    )
    
    # Draw plot showing residuals
    plt.style.use(['ticks_outtie'])
    
    fig, axs = plt.subplots(
        nrows=2, ncols=2,
        figsize=(20,6),
        frameon=True, sharex=True,
    )
    
    axs[0][0].plot(
        epoch_jyears,
        resids_abs_leg,
        'o', alpha=0.75,
        color='C0', label='Legacy',
    )
    
    axs[0][0].plot(
        epoch_jyears[hl_filt],
        resids_abs_leg[hl_filt],
        '.', alpha=0.75,
        color='w',
    )
    
    axs[0][0].plot(
        epoch_jyears,
        resids_abs_sin,
        'o', alpha=0.75,
        color='C1', label='Single-PSF',
    )
    
    axs[0][0].plot(
        epoch_jyears[hl_filt],
        resids_abs_sin[hl_filt],
        '.', alpha=0.75,
        color='w',
    )
    
    # axs[0][0].set_xlabel('Observation Date')
    axs[0][0].set_ylabel(r'Median $\sqrt{(data - model)^2}$')
    
    axs[0][0].legend(
        loc='upper right',
        ncol=1, fontsize='x-small',
        # bbox_to_anchor=(0, -0.27 1, 0.05), mode='expand',
    )
    
    axs[0][0].set_xlim([2006, 2024])
    axs[0][0].set_ylim([-50, 1100])
    
    axs[0][0].set_title("Entire Image")
    
    x_majorLocator = MultipleLocator(4)
    x_minorLocator = MultipleLocator(1)
    axs[0][0].xaxis.set_major_locator(x_majorLocator)
    axs[0][0].xaxis.set_minor_locator(x_minorLocator)

    y_majorLocator = MultipleLocator(200)
    y_minorLocator = MultipleLocator(50)
    axs[0][0].yaxis.set_major_locator(y_majorLocator)
    axs[0][0].yaxis.set_minor_locator(y_minorLocator)
    
    axs[1][0].plot(
        epoch_jyears,
        resids_abs_feu_leg,
        'o', alpha=0.75,
        color='C0', label='Legacy',
    )
    
    axs[1][0].plot(
        epoch_jyears[hl_filt],
        resids_abs_feu_leg[hl_filt],
        '.', alpha=0.75,
        color='w',
    )
    
    axs[1][0].plot(
        epoch_jyears,
        resids_abs_feu_sin,
        'o', alpha=0.75,
        color='C1', label='Single-PSF',
    )
    
    axs[1][0].plot(
        epoch_jyears[hl_filt],
        resids_abs_feu_sin[hl_filt],
        '.', alpha=0.75,
        color='w',
    )
    
    
    axs[1][0].set_xlabel('Observation Date')
    axs[1][0].set_ylabel(r'Median $[\sqrt{(data-model)^2}$ / $data]$')
    
    # axs[1][0].legend(
    #     loc='upper left',
    #     ncol=1, fontsize='x-small',
    #     # bbox_to_anchor=(0, -0.27 1, 0.05), mode='expand',
    # )
    
    axs[1][0].set_xlim([2006, 2024])
    axs[1][0].set_ylim([-0.1, 2.5])
    
    x_majorLocator = MultipleLocator(4)
    x_minorLocator = MultipleLocator(1)
    axs[1][0].xaxis.set_major_locator(x_majorLocator)
    axs[1][0].xaxis.set_minor_locator(x_minorLocator)
    
    y_majorLocator = MultipleLocator(0.5)
    y_minorLocator = MultipleLocator(0.1)
    axs[1][0].yaxis.set_major_locator(y_majorLocator)
    axs[1][0].yaxis.set_minor_locator(y_minorLocator)
    
    axs[0][1].plot(
        epoch_jyears,
        resids_abs_leg_ca,
        'o', alpha=0.75,
        color='C0', label='Legacy',
    )
    
    axs[0][1].plot(
        epoch_jyears[hl_filt],
        resids_abs_leg_ca[hl_filt],
        '.', alpha=0.75,
        color='w',
    )
    
    axs[0][1].plot(
        epoch_jyears,
        resids_abs_sin_ca,
        'o', alpha=0.75,
        color='C1', label='Single-PSF',
    )
    
    axs[0][1].plot(
        epoch_jyears[hl_filt],
        resids_abs_sin_ca[hl_filt],
        '.', alpha=0.75,
        color='w',
    )
    
    # axs[0][1].set_xlabel('Observation Date')
    axs[0][1].set_ylabel(r'Median $\sqrt{(data - model)^2}$')
    
    axs[0][1].legend(
        loc='upper right',
        ncol=1, fontsize='x-small',
        # bbox_to_anchor=(0, -0.27 1, 0.05), mode='expand',
    )
    
    axs[0][1].set_xlim([2006, 2024])
    axs[0][1].set_ylim([-50, 1100])
    
    axs[0][1].set_title("Central Arcsecond")
    
    
    x_majorLocator = MultipleLocator(4)
    x_minorLocator = MultipleLocator(1)
    axs[0][1].xaxis.set_major_locator(x_majorLocator)
    axs[0][1].xaxis.set_minor_locator(x_minorLocator)
    
    y_majorLocator = MultipleLocator(200)
    y_minorLocator = MultipleLocator(50)
    axs[0][1].yaxis.set_major_locator(y_majorLocator)
    axs[0][1].yaxis.set_minor_locator(y_minorLocator)
    
    axs[1][1].plot(
        epoch_jyears,
        resids_abs_feu_leg_ca,
        'o', alpha=0.75,
        color='C0', label='Legacy',
    )
    
    axs[1][1].plot(
        epoch_jyears[hl_filt],
        resids_abs_feu_leg_ca[hl_filt],
        '.', alpha=0.75,
        color='w',
    )
    
    axs[1][1].plot(
        epoch_jyears,
        resids_abs_feu_sin_ca,
        'o', alpha=0.75,
        color='C1', label='Single-PSF',
    )
    
    axs[1][1].plot(
        epoch_jyears[hl_filt],
        resids_abs_feu_sin_ca[hl_filt],
        '.', alpha=0.75,
        color='w',
    )
    
    axs[1][1].set_xlabel('Observation Date')
    axs[1][1].set_ylabel(r'Median $[\sqrt{(data-model)^2}$ / $data]$')
    
    # axs[1][1].legend(
    #     loc='upper left',
    #     ncol=1, fontsize='x-small',
    #     # bbox_to_anchor=(0, -0.27 1, 0.05), mode='expand',
    # )
    
    axs[1][1].set_xlim([2006, 2024])
    axs[1][1].set_ylim([-0.1, 2.5])
    
    x_majorLocator = MultipleLocator(4)
    x_minorLocator = MultipleLocator(1)
    axs[1][1].xaxis.set_major_locator(x_majorLocator)
    axs[1][1].xaxis.set_minor_locator(x_minorLocator)
    
    y_majorLocator = MultipleLocator(0.5)
    y_minorLocator = MultipleLocator(0.1)
    axs[1][1].yaxis.set_major_locator(y_majorLocator)
    axs[1][1].yaxis.set_minor_locator(y_minorLocator)
    
    
    fig.tight_layout()

    fig.savefig(out_name + f'_resids.pdf')
    fig.savefig(out_name + f'_resids.png', dpi=200)

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


resid_tables = {}
resid_centarcsec_tables = {}

kp_epochs_complete = []
h_epochs_complete = []

kp_jyear = []
h_jyear = []

for epoch_row in tqdm(epochs_table):
    epoch_combo_name = epoch_row['epoch'] + '_' + epoch_row['filter']
    
    # Check if histogram file exists
    hist_file = f'../{epoch_combo_name}/resid_stats_plots/resids.h5'
    
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
    cur_resid_table = Table.read(
        hist_file, format='hdf5', path='data',
    )
    
    hist_file = f'../{epoch_combo_name}/resid_stats_plots/resids_centarcsec.h5'
    
    cur_resid_centarcsec_table = Table.read(
        hist_file, format='hdf5', path='data',
    )
    
    resid_tables[epoch_combo_name] = cur_resid_table
    resid_centarcsec_tables[epoch_combo_name] = cur_resid_centarcsec_table

# Draw plots and make tables
construct_combined_hist_table(
    kp_epochs_complete, np.array(kp_jyear),
    resid_tables, resid_centarcsec_tables,
    'kp', 'kp',
)

construct_combined_hist_table(
    h_epochs_complete, np.array(h_jyear),
    resid_tables, resid_centarcsec_tables,
    'h', 'h',
)

construct_combined_hist_table(
    kp_epochs_complete + h_epochs_complete, np.array(kp_jyear + h_jyear),
    resid_tables, resid_centarcsec_tables,
    'allbands', 'allbands',
)