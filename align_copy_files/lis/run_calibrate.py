#!/usr/bin/env python

from kai.reduce import calibrate

starlists = ['mag20200707nirc2_kp_0.8_stf_v2_1.lis',
             'mag20200707nirc2_kp_0.8_stf_v3_0.lis']

for starlist in starlists:
    cal_stars = 'irs16NW,S3-22,S1-17,S1-34,S4-3,S1-1,S1-21,S3-370,S3-88,S3-36,S2-63'
    align_stars = 'irs16C,irs16NW,irs16CC'
    cal_first_star = 'irs16C'
    
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
