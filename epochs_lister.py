#!/usr/bin/env python

# Gather list of every epoch that has both legacy and Single PSF STF run
# ---
# Abhimat Gautam

from glob import glob
from astropy.table import Table
import astropy.table as table

import numpy as np

exclude_epochs = ['20050731nirc2', '20060502nirc2',
                  '20140805nirc2']

def construct_epoch_table(dr_path = '/g/ghez/data/dr/dr1',
        legacy_version_str = 'v2_3', single_version_str = 'v3_1', filt = 'kp'):
    legacy_search_str = '{0}/starlists/combo/*/starfinder_{1}/mag*{2}*stf.lis'.format(
                            dr_path, legacy_version_str, filt)

    legacy_starlists = glob(legacy_search_str)

    single_search_str = '{0}/starlists/combo/*/starfinder_{1}/mag*{2}*stf.lis'.format(
                            dr_path, single_version_str, filt)

    single_starlists = glob(single_search_str)

    prefix_str = '{0}/starlists/combo/'.format(dr_path)

    legacy_epoch_names = []
    for starlist in legacy_starlists:
        trimmed_str = starlist[len(prefix_str):]
    
        suffix_index = trimmed_str.index('/starfinder')
    
        epoch_name = trimmed_str[0:suffix_index]
    
        legacy_epoch_names.append(epoch_name)


    single_epoch_names = []
    for starlist in single_starlists:
        trimmed_str = starlist[len(prefix_str):]
    
        suffix_index = trimmed_str.index('/starfinder')
    
        epoch_name = trimmed_str[0:suffix_index]
    
        single_epoch_names.append(epoch_name)


    # Check for common epochs
    common_epochs_filt = np.isin(legacy_epoch_names, single_epoch_names)

    common_epochs = (np.array(legacy_epoch_names))[common_epochs_filt]
    
    common_epochs_cleaned = []
    for epoch in common_epochs:
        if epoch in exclude_epochs:
            continue
        common_epochs_cleaned.append(epoch)
    
    common_epochs = common_epochs_cleaned

    # Go through epochs to figure out which are multi night combos
    combo_epochs_column = []
    filt_column = []
    
    for epoch in common_epochs:
        filt_column.append(filt)
        if epoch.find('_') == -1:
            combo_epochs_column.append('single_night')
        else:
            combo_epochs_column.append('multi_night')
    
    # Determine epoch quality statistics from combo log file
    num_frames = []
    med_fwhms = []
    med_strehls = []
    
    for epoch in common_epochs:
        log_file = f'{dr_path}/combo/{epoch}/mag{epoch}_{filt}.log'
        log_table = Table.read(log_file, format='ascii')
        
        num_frames.append(len(log_table))
        med_fwhms.append(np.median(log_table['col2']))
        med_strehls.append(np.median(log_table['col3']))
        
    
    # Construct final Astropy table
    epochs_table = Table({'epoch': common_epochs,
                          'filt': filt_column,
                          'nights_combo': combo_epochs_column,
                          'num_frames': num_frames,
                          'med_fwhms': med_fwhms,
                          'med_strehls': med_strehls})
    return epochs_table


epochs_table_kp = construct_epoch_table(dr_path = '/g/ghez/data/dr/dr1',
                        legacy_version_str = 'v2_3',
                        single_version_str = 'v3_1',
                        filt = 'kp')

epochs_table_h = construct_epoch_table(dr_path = '/g/ghez/data/dr/dr1',
                        legacy_version_str = 'v2_3',
                        single_version_str = 'v3_1',
                        filt = 'h')

epochs_table = table.vstack([epochs_table_kp, epochs_table_h])
epochs_table.sort('epoch')

epochs_table.write('epochs_table.txt', format='ascii.fixed_width_two_line',
                   overwrite=True)

epochs_table.write('epochs_table.h5', format='hdf5',
                   path='data', serialize_meta=True,
                   overwrite=True)

print(epochs_table)