import os 
import sys
import math

import numpy as np 
import pandas as pd

from datetime import datetime, timedelta

def correct_rotation(a):
    #s0 = output_varr[i,0]
    #s1 = output_varr[i,1]
    
    s0 = a[:,0]
    s1 = a[:,1]
    tc_angle = a[:,2]
    
    x = np.cos(s1)*np.sin(s0) # -40*
    y = np.sin(s1) # -40*
    z = -np.cos(s1)*np.cos(s0)
    
    tangle = -(tc_angle/180)*np.pi
    xt = np.cos(tangle)*x + np.sin(tangle)*y
    yt = -np.sin(tangle)*x + np.cos(tangle)*y
    zt = z
    
    vt = np.stack((xt,yt,zt), 1)
    #vt = np.array([xt,yt,zt])
    vtnorm = (vt*vt).sum(1)
    vt = vt / np.sqrt(vtnorm.reshape(-1,1))
    
    ns0 = np.arctan2(vt[:,0],-vt[:,2])
    ns1 = np.arcsin(vt[:,1])
    
    return np.stack((ns0,ns1,tc_angle), 1)

def eval_thrshld(phi, theta, gt_lab, lims):
	gt_gaze = gt_lab.astype(np.int32)
	
	phi_min = phi > (lims[0]/180.0)*math.pi #-0.9
	phi_min_dist= phi - (lims[0]/180.0)*math.pi
	phi_max = phi < (lims[1]/180.0)*math.pi #0.9
	phi_max_dist= (lims[1]/180.0)*math.pi - phi
	
	phi_constraint_met = np.logical_and(phi_min, phi_max)
	
	phi_min_dist[phi_min_dist>0] = 0
	phi_min_dist = -1*phi_min_dist
	
	phi_max_dist[phi_max_dist>0] = 0
	phi_max_dist = -1*phi_max_dist
	phi_dist = phi_min_dist + phi_max_dist
	
	the_min = theta > (lims[2]/180.0)*math.pi #-.75
	the_min_dist = theta - (lims[2]/180.0)*math.pi
	the_max = theta < (lims[3]/180.0)*math.pi #0.1
	the_max_dist = (lims[3]/180.0)*math.pi - theta #0.1
	
	the_min_dist[the_min_dist>0] = 0
	the_min_dist = -1*the_min_dist

	the_max_dist[the_max_dist>0] = 0
	the_max_dist = -1*the_max_dist
	the_dist = the_min_dist + the_max_dist
	
	the_constraint_met = np.logical_and(the_min, the_max)	
	
	
	phi_theta_met = np.logical_and(the_constraint_met, phi_constraint_met)
	gaze_est = np.array(phi_theta_met)*1
	gaze_est = gaze_est.astype(np.int32)
	geval = gaze_est == gt_gaze
	acc = np.array(geval).sum()/float(phi.shape[0])*100
	
	return acc, gaze_est, phi_dist, the_dist

H = 342; W = 608
pH = 35 #35 # 57 # 21
pW = 53 #53 # 100 # 31
eH = 7  #7 # 17 # 3
eW = 9  #9 # 23 # 3

xH = 2
xW = 2

x, y = np.meshgrid(np.arange(0,W,pW), np.arange(0,H,pH)) # y varies along col, x varies along row
y = y.reshape(-1)
x = x.reshape(-1)

def convert_lims(tv_data):

    #load lims
    loc_lims = np.load('C:\\Users\\u255769\\flash-post-processing\\Evaluation\\4331_v3r50reg_reg_testlims_35_53_7_9.npy')
    loc_lims = loc_lims.reshape(-1,4)

    drl = (loc_lims[:,1]-loc_lims[:,0])/2.0
    dtb = (loc_lims[:,3]-loc_lims[:,2])/2.0

    tbshift = 0 
    lrshift = 0 
    slr = 1.1
    stb = 1.1
    rls_sc = 0.0
    tbs_sc = 0.0
    
    tv_size = tv_data['size']
    tv_height = tv_data['tv_height']
    cam_height = tv_data['cam_height']
    
    fam_ = 'center-'
    # mru tv - 50 inches
    # apple max - 30 inches
    if tv_size < 37:
        tvs = 'small'
    else:
        tvs = 'big'
    fam_ += tvs+'-med'
    
    cam_below_tv = False
    if cam_height <= tv_height: 
        cam_below_tv = True
    elif cam_height <= tv_height+5:
        cam_below_tv = True

    if cam_below_tv:
        tb_shift = 15 if cam_height <= 35 else 10
    else:
        if cam_height >= 60:
            tb_shift = -15
        elif cam_height >= 50:
            tb_shift = -10
        else:
            tb_shift = 0
            
    #fam_ = 'center-big-med' #start_time_list[famid]
    fam_ = fam_.split('-')
    pos = fam_[0] # center, left, right
    size = fam_[1] # small, big
    height = fam_[2] # med, max, min

    if size=='big':
        rls_sc = 0.3
        tbs_sc = 0.2
        stb = 1.1
    else:
        rls_sc = 0.1
        tbs_sc = 0.05
	
    if pos=='center':
        slr = 1.1
         
    if pos=='left':
         slr = 1.35
         
    if pos=='right':
        slr = 1.35
    
    if pos!='center' and size=='small':
        rls_sc = 0.2
        tbs_sc = 0.1 #0.1

    rls = drl*rls_sc
    tbs = dtb*tbs_sc

    #max height, tv large corners	bottom lims -10
    #max height, tv large center	bottom lims -7
    #least height, tv large	bottom lims -3
    #tbshift=10 #15
    #lrshift=0

    loc_lims[:,0] = slr*(loc_lims[:,0]) -rls + lrshift # left right lower lim and upper lim
    loc_lims[:,1] = slr*(loc_lims[:,1]) +rls + lrshift 

    loc_lims[:,2] = stb*(loc_lims[:,2]) -tbs + tbshift # top bottom lower lim upper lim
    loc_lims[:,3] = stb*(loc_lims[:,3]) +tbs + tbshift
    
    return loc_lims

def convert_to_gaze(at, loc_lims):

    # ['phi','theta','top','left','bottom','right']
    pred_gz_data = 2*np.ones((at.shape[0]))
    gt_gz = np.zeros((at.shape[0]))
    gt_gz = gt_gz.reshape(-1,1)
    
    for i in range(y.shape[0]):

        leftl = max(x[i],0)
        rightl = min(x[i]+pW+xW, 608)
        topl = max(y[i], 0)
        bottoml = min(y[i]+pH+xH, 342)

        at[:, 3] = pd.to_numeric(at[:, 3], errors='coerce')
        at[:, 5] = pd.to_numeric(at[:, 5], errors='coerce')

        colm = (at[:,3] + at[:,5])/2
        rowm = (at[:,2] + at[:,4])/2
	    
        frame_filter_col = np.logical_and(colm>leftl, colm<rightl)
        frame_filter_row = np.logical_and(rowm>topl, rowm<bottoml)
        frame_filter = np.logical_and(frame_filter_col,frame_filter_row)
        
        rows = leftl
        cols = topl
        rowe = rightl
        cole = bottoml

        lims = loc_lims[i]
        Ntest = (frame_filter).sum()	
        
        if Ntest>=1:
            phi_loc = at[frame_filter,0]
            theta_loc = at[frame_filter,1]
            gaze_loc = gt_gz[frame_filter,0]
            
            if any([math.isnan(l) for l in lims]):
                print(rows, rowe, cols, cole)
                print('got to skip', 'test samp', Ntest, 'gz', gaze_loc.sum(), 'no gz', gaze_loc.shape[0]-gaze_loc.sum())
                continue
            
            acc, ge,_,_ = eval_thrshld(phi_loc, theta_loc, gaze_loc, lims)
            if phi_loc.size != ge.size:
                print('size diff:', phi_loc.shape, ge.shape)
                
            pred_gz_data[frame_filter] = ge
    
    return pred_gz_data