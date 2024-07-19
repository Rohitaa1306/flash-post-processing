import os 
import sys
import random
import math
import argparse
import time

import yaml 

import numpy as np 

from pathlib import Path
from utils import * 
from gaze_utils import * 
from tv_power import *

#tv_data = {'id': 6, 'location': 'living room', 
#                'measure': {'size': 51.0, 'cam_height': 43.0, 'tv_height': 61.0, 'view_dist': 164.0}, 
#                'config': {'device_id': 6, 'power_read': False, 'power_threshold': 10, 'powered_down': False, 'powered_down_data': 'None'}
#            }

def main(base_path, ppt_id, start_date, tv_data, num_days=10, study4=False):
    
    prefix_ = 'P1-' if not study4 else ''
    
    dev_id = int(tv_data['id'])
    txt_path = os.path.join(base_path, ppt_id, '%s%s%s_data'%(prefix_, ppt_id, str(dev_id).zfill(3)), 'txts')
    power_path = os.path.join(base_path, ppt_id, '%s%s%s_data'%(prefix_, ppt_id, str(dev_id).zfill(3)), '%s%s%s_tv_power_5s.csv'%(prefix_, ppt_id, str(dev_id).zfill(3)))
    #'/home/akv/FLASH_PO1/techData/1147/P1-1147011_data/P1-1147011_tv_power_5s.csv'
    print(txt_path)
    print(power_path)
    
    power_down = tv_data['config']['powered_down'] #False
    power_down_data = None
    if power_down:
        power_down_data = tv_data['config']['powered_down_data']
    
    tv_power_read = tv_data['config']['power_read']
    tv_power_threshold = tv_data['config']['power_threshold']
    #num_days = 10
    mixmodel = True

    day_start = '00:00:00'
    day_end = '23:59:59'
    day_end_epc = '23:59:55'
    datetime_format = "%Y-%m-%d %H:%M:%S.%f"    # Datetime format corresponding to the datetime string

    date_match, txt_date_ts = sort_files(txt_path, start_date, num_days)
    date_keys = list(date_match.keys())
    print('End Date \t: %s'%date_keys[-1]) 

    # Read power logs
    if tv_power_read:
        start_dt = str(date_keys[0]) + ' ' + '00:00:00'
        end_dt = str(date_keys[-1]) + ' ' + '23:59:55'

        series = read_power_pd(power_path)
        series = filter_series(series, start_dt, end_dt)
        power_df = interp_series(series, start_dt, end_dt)
        #print(power_df)

    miss_ts = create_dict(keys=date_match.keys())
    gz_epc_dfs = []
    
    dates_range = pd.date_range(start=start_date, periods=num_days, freq='D')
    summary_df = pd.DataFrame(index=dates_range, columns=['miss_time', 'gaze_epc', 'exp_epc', 'miss_tvon', 'gaze_tvon', 'exp_tvon'])
    summary_df.index.name = 'Date'
    #summary_df.loc[pd.to_datetime('2023-11-10'),'miss']=10.0

    lims = convert_lims(tv_data['measure'])
    for date_key in date_match.keys():
        #print('Processing date: ', str(date_key))

        # miss_time, ts
        miss_time, ts_list, txt_dfs = get_miss_time(txt_path, date_key, file_ls=date_match[date_key])
        summary_df.loc[pd.to_datetime(date_key), 'miss_time'] = miss_time/60.0

        miss_ts[date_key]['time'] = miss_time 
        miss_ts[date_key]['ts_list'] = ts_list
        #print(ts_list)
        
        #print('Missed time (mins): \t%.2f'%(miss_time/60.0))
        
        # read reg and rot files    
        file_ls = [file_.replace('.txt','_rot.txt') for file_ in date_match[date_key]]
        rot_dfs = get_df_date(txt_path, file_ls, date_key)
        rot_len = [df.shape[0] for df in rot_dfs]
        
        file_ls = [file_.replace('.txt','_reg.txt') for file_ in date_match[date_key]]
        reg_dfs = get_df_date(txt_path, file_ls, date_key)
        reg_len = [df.shape[0] for df in reg_dfs]
        
        #print(file_ls)
        #print(reg_len, rot_len)
        assert reg_len==rot_len    
        for i, l in enumerate(reg_len):
            if l==0:
                assert not any(txt_dfs[i]['tag']=='Gaze-det')
        
        # gaze data frame
        start_date = str(date_key) + ' ' + day_start
        end_date = str(date_key) + ' ' + day_end
        end_date_ts = convert_to_datetime(date_string=end_date, date_format="%Y-%m-%d %H:%M:%S")
        
        gz_index = pd.date_range(start=start_date, end=end_date, freq='S')
        gz_df = pd.DataFrame(index=gz_index)
        gz_df['TC_gaze'] = 5
        gz_df['TC_exposure_only'] = 5
        
        #['dateTimeStamp', 'frameNum', 'numFaces', 'tcPresent', 'phi', 'theta', 'sigma', 'rot.', 'top', 'left', 'bottom', 'right', 'tag']
        for i, rot_df in enumerate(rot_dfs):
            rot_df.set_index('dateTimeStamp', inplace=True)
            rot_df = rot_df.sort_values(by='dateTimeStamp')
            #print(rot_df.shape[0])
            rot_df.index = rot_df.index.floor('S')
            
            reg_df = reg_dfs[i]
            reg_df.set_index('dateTimeStamp', inplace=True)
            reg_df = reg_df.sort_values(by='dateTimeStamp')
            reg_df.index = reg_df.index.floor('S')
            
            index_match = rot_df.index.equals(reg_df.index)
            assert index_match
            
            # correct for rotation for phi and theta
            df_ = reg_df[reg_df['rot.'].abs()>=30] 
            if df_.shape[0]>0:
                phi_ = df_[['phi','theta','rot.']].values #['phi','theta','rot.'] 
                phi_corrected = correct_rotation(phi_)
                reg_df.loc[df_.index,['phi','theta']] = phi_corrected[:,:2]
            

            #0 = No-Gz (flash pred) Gaze-no-det, TC not present    
            df_ = rot_df[rot_df['tag']=='Gaze-no-det']
            assert df_['tcPresent'].sum()==0
            
            gz_df.loc[df_.index,'TC_gaze'] = 0
            inc_index = df_.index + pd.Timedelta(seconds=1)
            inc_index = inc_index[inc_index<=end_date_ts] # prevents it from spilling over to the next date
            gz_df.loc[inc_index,'TC_gaze'] = 0 # cause FLASHtv takes frames 2-sec and gives one prediction
            
            gz_df.loc[df_.index,'TC_exposure_only'] = 0
            gz_df.loc[inc_index,'TC_exposure_only'] = 0 # cause FLASHtv takes frames 2-sec and gives one prediction
            
            del df_
            
            #1 = Gz (flash pred) Gaze-det, TC present
            #0 = No-Gz (flash pred) Gaze-det, TC present (TC exposure only)
            df_1 = rot_df[rot_df['tag']=='Gaze-det']
            df_2 = reg_df[rot_df['tag']=='Gaze-det']
            assert len(df_1)==len(df_2)
            
            gz_data = df_1[['phi','theta','top','left','bottom','right']].values
            gz_reg = df_1[['phi','theta']].values

            if mixmodel:
                gz_data[:,:2] = 0.6*gz_data[:,:2] + 0.4*gz_reg
            
            pred_gz = convert_to_gaze(gz_data, lims).astype(np.int32)
            tc_exp_only = 1 - pred_gz
            
            assert pred_gz.shape[0] == gz_data.shape[0]
            assert (pred_gz==2).sum() == 0
            assert (pred_gz<0).sum() == 0
            assert (pred_gz>1).sum() == 0
            
            gz_df.loc[df_1.index,'TC_gaze'] = pred_gz
            inc_index = df_1.index + pd.Timedelta(seconds=1)
            bool_index = inc_index<=end_date_ts
            inc_index = inc_index[inc_index<=end_date_ts] # prevents it from spilling over to the next date

            gz_df.loc[inc_index,'TC_gaze'] = pred_gz[bool_index]

            gz_df.loc[df_1.index,'TC_exposure_only'] = tc_exp_only
            gz_df.loc[inc_index,'TC_exposure_only'] = tc_exp_only[bool_index]
                
        #2 = FLASH miss time (reboot, restart or error)
        #3 = FLASH powered down
        
        #2 = FLASH miss time (reboot, restart or error)
        for ts_miss in miss_ts[date_key]['ts_list']:
            #print(ts_miss)
            start_miss, end_miss = ts_miss
            start_miss, end_miss = start_miss.floor('S'), end_miss.floor('S')
            miss_index = pd.date_range(start=start_miss, end=end_miss, freq='S')
            
            gz_df.loc[miss_index,'TC_gaze'] = 2
            gz_df.loc[miss_index,'TC_exposure_only'] = 2
        
        # Define the start and end dates of the powered down on a specific date
        if power_down:
            
            for pd_data in power_down_data:
                start_miss = datetime.strptime(pd_data[0],"%Y-%m-%d %H:%M:%S.%f")
                end_miss = datetime.strptime(pd_data[1],"%Y-%m-%d %H:%M:%S.%f")
                
                if date_key==start_miss.date():
                    start_miss = pd.to_datetime(start_miss)
                    end_miss = pd.to_datetime(end_miss)
                    start_miss, end_miss = start_miss.floor('S'), end_miss.floor('S')

                    # Create a time series index with intervals of one second
                    p_index = pd.date_range(start=start_miss, end=end_miss, freq='S')
                    
                    gz_df.loc[p_index,'TC_gaze'] = 3
                    gz_df.loc[p_index,'TC_exposure_only'] = 3
            
        # flash per sec tv data
        gz_df[gz_df==5] = 0
        tv_data = gz_df[['TC_gaze','TC_exposure_only']].values
        tv_time_sec = (tv_data[:,0]==1).sum()/60.0
        tv_exp_only_sec = (tv_data[:,1]==1).sum()/60.0
        
        #mic_check = (tv_data[:,0]==2).sum()/60.0
        #print('TV time persec: \t%.2f'%tv_time_sec)
        #print('TV exponly persec: \t%.2f'%tv_exp_only_sec)
        #print('TV miss persec: \t%.2f'%mic_check)
        
        tv_time = tv_data[:,0]
        tv_exp_only = tv_data[:,1]
        tv_time_epc, tv_exp_only_epc = condense_epc(tv_time, tv_exp_only)
        
        #mic_check = (tv_time_epc==2).sum()*5/60.0
        #print('TV miss perepc: \t%.2f'%mic_check)
        
        # flash per epoch tv data
        gz_epoch_index = pd.date_range(start=start_date, end=end_date, freq='5S')
        gz_epc_df = pd.DataFrame(index=gz_epoch_index)
        gz_epc_df.loc[gz_epc_df.index,'TC_gaze'] = tv_time_epc
        gz_epc_df.loc[gz_epc_df.index,'TC_exposure_only'] = tv_exp_only_epc
        
        tt = (tv_time_epc==1).sum()*5/60.0
        eo = (tv_exp_only_epc==1).sum()*5/60.0
        
        #print('TV time perepc: \t%.2f'%tt)
        #print('TV exponly perepc: \t%.2f'%eo)
        
        summary_df.loc[pd.to_datetime(date_key), 'gaze_epc'] = tt
        summary_df.loc[pd.to_datetime(date_key), 'exp_epc'] = eo
        
        
        # combine with the TV power data 
        if tv_power_read:
            start_epc = str(date_key) + ' ' + day_start
            end_epc = str(date_key) + ' ' + day_end_epc
            tvpower_ = power_df[pd.to_datetime(start_epc):pd.to_datetime(end_epc)]
            tvpower = tvpower_['power'].values
            
            #tv_power_threshold
            tvpower = 1*(tvpower > tv_power_threshold)
            
            tv_time_epc = tv_time_epc * tvpower
            tv_exp_only_epc = tv_exp_only_epc * tvpower

            tt = (tv_time_epc==1).sum()*5/60.0
            eo = (tv_exp_only_epc==1).sum()*5/60.0
            mo = (tv_time_epc==2).sum()*5/60.0
            mo = mo + (tv_time_epc==3).sum()*5/60.0
            
            #print('TV time epc tv: \t%.2f'%tt)
            #print('TV exponly epc tv: \t%.2f'%eo)

            gz_epc_df.loc[gz_epc_df.index,'TC_gaze'] = tv_time_epc
            gz_epc_df.loc[gz_epc_df.index,'TC_exposure_only'] = tv_exp_only_epc
            
            summary_df.loc[pd.to_datetime(date_key), 'gaze_tvon'] = tt
            summary_df.loc[pd.to_datetime(date_key), 'exp_tvon'] = eo
            summary_df.loc[pd.to_datetime(date_key), 'miss_tvon'] = mo
        
        
        gz_epc_df.index.name = 'dateTimeStamp'
        #gz_epc_df.astype(int).to_csv(args.outcsv+'_'+str(date_key)+'.csv', sep=',')
        
        gz_epc_dfs.append(gz_epc_df)
    
    #summary_df = None
    return gz_epc_dfs, summary_df
        
if __name__ == "__main__":
    
    #txt_path = '/home/akv/FLASH_PO1/techData/1147/P1-1147011_data/txts2'
    #power_path = '/home/akv/FLASH_PO1/techData/1147/P1-1147011_data/P1-1147011_tv_power_5s.csv'
    
    base_path = '/home/akv/FLASH_PO1/techData'
    ppt_id = 1311
    ppt_id = str(ppt_id)
    start_date = '2024-01-12'
    
    tv_data = {'id': 6, 'location': 'Parent room', 
                'measure': {'size': 51.0, 'cam_height': 43.0, 'tv_height': 61.0, 'view_dist': 164.0}, 
                'config': {'device_id': 6, 'power_read': True, 'power_threshold': 25.0, 
                           'powered_down': False, 
                           'powered_down_data': None
                            }
            }

    gaze_dfs, summary_df = main(base_path, ppt_id, start_date, tv_data)
    
    #print(len(gaze_dfs))
    print(summary_df)
    #print(gaze_dfs[-1])


