#!/usr/bin/env python

# Setup and run aligns between starlists for every common epoch
# ---
# Abhimat Gautam

from astropy.table import Table

import numpy as np
import os
import shutil

from tqdm import tqdm

from kai.reduce import calibrate

dr_path = '/g/ghez/data/dr/dr1'
legacy_version_str = 'v2_3'
single_version_str = 'v3_1'
stf_corr = '0.8'

align_copy_path = './align_copy_files'

align_flags = '-r align_d -m 16 -a 0 -p -v -N ../source_list/label_abs.dat -o ../source_list/orbits.dat -O 2 align_d.list'
align_abs_flags = '-a 3 -abs ../source_list/absolute_refs.dat align_d align_d_abs '

def run_calibrate(starlist,
        cal_stars = 'irs16NW,S3-22,S1-17,S1-34,S4-3,S1-1,S1-21,S3-370,S3-88,S3-36,S2-63',
        align_stars = 'irs16C,irs16NW,irs16CC',
        cal_first_star = 'irs16C'):
    
    calibrate_args = ''

    calibrate_args += '-f 1 '
    calibrate_args += '-R -V '
    calibrate_args += '-N ../source_list/photo_calib.dat '
    calibrate_args += '-M Kp -T 0 '

    calibrate_args += '-S {0} '.format(cal_stars)
    calibrate_args += '-A {0} '.format(align_stars)
    calibrate_args += '-I {0} '.format(cal_first_star)

    calibrate_args += '-c 4 '

    calibrate_args += starlist
    
    print('calibrate ' + calibrate_args)
    
    calibrate.main(calibrate_args.split())


# Read in epochs table
epochs_table = Table.read('epochs_table.h5', format='hdf5', path='data')

epochs_table = epochs_table[np.where(epochs_table['nights_combo'] == 'single_night')]

# Go through and process each epoch
orig_wd = os.getcwd()

for epochs_row in tqdm(epochs_table):
    cur_epoch = epochs_row['epoch']
    cur_filt = epochs_row['filt']
    
    cur_epoch_align_dir = '{0}_{1}/starlists_align/'.format(cur_epoch, cur_filt)
    
    os.makedirs(cur_epoch_align_dir, exist_ok=True)
    
    # Copy source_list directory
    shutil.copytree('{0}/source_list'.format(align_copy_path),
                    '{0}/source_list'.format(cur_epoch_align_dir),
                    dirs_exist_ok=True)
    
    # Make lis directory and copy starlists
    lis_dir_loc = '{0}/lis/'.format(cur_epoch_align_dir)
    os.makedirs(lis_dir_loc, exist_ok=True)
    
    orig_leg_stf_file = '{0}/starlists/combo/{1}/starfinder_{2}/mag{1}_{3}_{4}_stf.lis'.format(
                            dr_path, cur_epoch, legacy_version_str, cur_filt, stf_corr)
    orig_sin_stf_file = '{0}/starlists/combo/{1}/starfinder_{2}/mag{1}_{3}_{4}_stf.lis'.format(
                            dr_path, cur_epoch, single_version_str, cur_filt, stf_corr)
    
    new_leg_stf_file = 'mag{0}_{1}_{2}_stf_{3}.lis'.format(
                            cur_epoch, cur_filt, stf_corr, legacy_version_str)
    new_sin_stf_file = 'mag{0}_{1}_{2}_stf_{3}.lis'.format(
                            cur_epoch, cur_filt, stf_corr, single_version_str)
    
    shutil.copy(orig_leg_stf_file, lis_dir_loc + new_leg_stf_file)
    shutil.copy(orig_sin_stf_file, lis_dir_loc + new_sin_stf_file)
    
    # Run calibrate
    os.chdir(lis_dir_loc)
    
    run_calibrate(new_leg_stf_file)
    run_calibrate(new_sin_stf_file)
    
    os.chdir(orig_wd)
    
    # Make align directory and align_d.list file
    align_dir_loc = '{0}/align/'.format(cur_epoch_align_dir)
    os.makedirs(align_dir_loc, exist_ok=True)
    
    cal_leg_stf_file = 'mag{0}_{1}_0.8_stf_{2}_cal.lis'.format(
                            cur_epoch, cur_filt, legacy_version_str)
    cal_sin_stf_file = 'mag{0}_{1}_0.8_stf_{2}_cal.lis'.format(
                            cur_epoch, cur_filt, single_version_str)
    
    list_file = ''
    list_file += '../lis/{0} 8 \n'.format(cal_leg_stf_file)
    list_file += '../lis/{0} 8 '.format(cal_sin_stf_file)
    
    with open(align_dir_loc + 'align_d.list', 'w') as out_file:
        out_file.write(list_file)
    
    # Run align
    os.chdir(align_dir_loc)
    
    java_align_command = 'java align ' + align_flags
    os.system(java_align_command)
    
    # java_align_abs_command = 'java align ' + align_abs_flags
    # os.system(java_align_abs_command)
    
    os.chdir(orig_wd)
