# File readers for starfinder comparison analyses
# ---
# Abhimat Gautam

import numpy as np
from astropy.table import Table

def stf_lis_reader(file_loc):
    # Read in table
    stf_table = Table.read(file_loc, format='ascii')
    
    if stf_table.colnames[0] == 'col1':
        # Rename columns
        stf_table.rename_column('col1', 'name')
        stf_table.rename_column('col2', 'mag')
        stf_table.rename_column('col3', 'epoch')
        stf_table.rename_column('col4', 'x')
        stf_table.rename_column('col5', 'y')
        stf_table.rename_column('col6', 'snr')
        stf_table.rename_column('col7', 'corr')
        stf_table.rename_column('col8', 'N_frames')
        stf_table.rename_column('col9', 'flux')
    
    return stf_table

def stf_rms_lis_reader(file_loc):
    # Read in table
    stf_table = Table.read(file_loc, format='ascii')
    
    if stf_table.colnames[0] == 'col1':
        # Rename columns
        stf_table.rename_column('col1', 'name')
        stf_table.rename_column('col2', 'm')
        stf_table.rename_column('col3', 't')
        stf_table.rename_column('col4', 'x')
        stf_table.rename_column('col5', 'y')
        stf_table.rename_column('col6', 'xe')
        stf_table.rename_column('col7', 'ye')
        stf_table.rename_column('col8', 'snr')
        stf_table.rename_column('col9', 'corr')
        stf_table.rename_column('col10', 'N_frames')
        stf_table.rename_column('col11', 'flux')
    
    # Add an additional column for error in mag
    stf_table['me'] = 1.0 / stf_table['snr']
    
    return stf_table


def align_orig_pos_reader(file_loc,
                          align_stf_1_version='v2_3',
                          align_stf_2_version='v3_1'):
    # Read in table
    align_orig_pos_table = Table.read(file_loc, format='ascii')
    
    # Rename columns
    align_orig_pos_table.rename_column('col1', 'name')
    align_orig_pos_table.rename_column('col2', align_stf_1_version + '_x')
    align_orig_pos_table.rename_column('col3', align_stf_1_version + '_y')
    align_orig_pos_table.rename_column('col4', align_stf_2_version + '_x')
    align_orig_pos_table.rename_column('col5', align_stf_2_version + '_y')
    
    return align_orig_pos_table

def align_pos_reader(file_loc,
                     align_stf_1_version='v2_3',
                     align_stf_2_version='v3_1'):
    # Read in table
    align_pos_table = Table.read(file_loc, format='ascii')
    
    # Rename columns
    align_pos_table.rename_column('col1', 'name')
    align_pos_table.rename_column('col2', align_stf_1_version + '_x')
    align_pos_table.rename_column('col3', align_stf_1_version + '_y')
    align_pos_table.rename_column('col4', align_stf_2_version + '_x')
    align_pos_table.rename_column('col5', align_stf_2_version + '_y')
    
    return align_pos_table

def align_pos_err_reader(file_loc,
                         align_stf_1_version='v2_3',
                         align_stf_2_version='v3_1'):
    # Read in table
    align_pos_err_table = Table.read(file_loc, format='ascii')
    
    # print(align_pos_err_table)
    
    # Rename columns
    align_pos_err_table.rename_column('col1', 'name')
    align_pos_err_table.rename_column('col2', align_stf_1_version + '_x_err')
    align_pos_err_table.rename_column('col3', align_stf_1_version + '_y_err')
    align_pos_err_table.rename_column('col6', align_stf_2_version + '_x_err')
    align_pos_err_table.rename_column('col7', align_stf_2_version + '_y_err')
    
    # print(align_pos_err_table)
    
    align_pos_err_table.remove_columns(['col4', 'col5',
                                        'col8', 'col9'])
    
    # print(align_pos_err_table)
    
    return align_pos_err_table

def align_mag_reader(file_loc,
                     align_stf_1_version='v2_3',
                     align_stf_2_version='v3_1'):
    # Read in table
    align_mag_table = Table.read(file_loc, format='ascii')
    
    # Rename columns
    align_mag_table.rename_column('col1', 'name')
    align_mag_table.rename_column('col2', 'avg_mag')
    
    align_mag_table.rename_column('col5', align_stf_1_version + '_mag')
    align_mag_table.rename_column('col6', align_stf_2_version + '_mag')
    
    return align_mag_table
