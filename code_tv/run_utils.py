import os 
import sys
import random

import numpy as np 
import pandas as pd

from datetime import datetime, timedelta

datetime_format = "%Y-%m-%d %H:%M:%S.%f"    # Datetime format corresponding to the datetime string

# random check 
def rand_day_check(start_date, df, num_days=10):

    # Number of days to add
    days_to_add = random.randint(0,num_days-1)
    given_date = pd.to_datetime(start_date)
    target_date = given_date + pd.Timedelta(days=days_to_add)
    target_end_date = target_date + pd.Timedelta(hours=23,minutes=59,seconds=55)

    # Generate datetime index with 5-second interval for the target date
    #date_range = pd.date_range(start=target_date, periods=24*60*12, freq='5S')
    print('Check date \t: %s'%str(target_date))
    #print('Check date \t: %s'%str(target_end_date))
    
    #['tv1-gaze', 'tv1-exponly', 'tv2-gaze', 'tv2-exponly', 'tv3-gaze', 'tv3-exponly']
    df_date = df[target_date:target_end_date]
    print('verify the =1 sum with the summary table')
    print((df_date==1).sum()/12.0)
    
    print('verify the =2 sum with the summary table')
    print((df_date==2).sum()/12.0)
    
    return None

def get_tv_data(ppt_data, tv_config, tv_count):
    dev_ids = []
    tvdev_data = {}
    for i in range(tv_count):
        i = i+1
        
        dict_ = {} # devid, location, meeasurements
        dict_['id'] = ppt_data.iloc[0]['flash'+str(i)]
        dev_id = dict_['id']
        if np.isnan(dict_['id']): 
            tv_count = i-1
            break
        
        dict_['location'] = ppt_data.iloc[0]['loc'+str(i)]
        dict_['measure'] = {
                            'size': ppt_data.iloc[0]['length'+str(i)],
                            'cam_height': ppt_data.iloc[0]['camheight'+str(i)],
                            'tv_height': ppt_data.iloc[0]['bottom'+str(i)],
                            'view_dist': ppt_data.iloc[0]['viewdistance'+str(i)],
                           }
        for key in tv_config.keys():
            if tv_config[key]['device_id'] == dev_id:
                dict_['config'] = tv_config[key]
                
        tvdev_data[i-1] = dict_
        dev_ids.append(int(dev_id))
    
    return tvdev_data, dev_ids, tv_count 

def get_tv_data_study4(tv_config_old, tv_count):
    dev_ids = []
    tvdev_data = {}
    for i in range(tv_count):
        i = i+1
        
        dict_ = {} # devid, location, meeasurements
        dict_['id'] = tv_config_old['tv'+str(i)]['device_id']
        dev_id = dict_['id']
        if np.isnan(dict_['id']): 
            #tv_count = i-1
            break
        
        dict_['location'] = tv_config_old['tv'+str(i)]['location']
        dict_['measure'] = {
                            'size': tv_config_old['tv'+str(i)]['size'],
                            'cam_height': tv_config_old['tv'+str(i)]['cam_height'],
                            'tv_height': tv_config_old['tv'+str(i)]['tv_height'],
                            'view_dist': tv_config_old['tv'+str(i)]['dist']
                           }
        #for key in tv_config.keys():
        #    if tv_config[key]['device_id'] == dev_id:
        dict_['config'] = tv_config_old['tv'+str(i)]
                
        tvdev_data[i-1] = dict_
        dev_ids.append(int(dev_id))
    
    return tvdev_data, dev_ids, tv_count 

def combine_gaze_tvs(gaze_df_tvs, tv_count, max_num_tvs=3):
    assert len(gaze_df_tvs) == tv_count
    
    combined_data = pd.DataFrame(index=gaze_df_tvs[0].index, columns=['tv1-gaze', 'tv1-exponly', 'tv2-gaze', 'tv2-exponly', 'tv3-gaze', 'tv3-exponly'], data=-1)
    for i, df in enumerate(gaze_df_tvs):
        if df is not None:
            combined_data.loc[df.index,'tv%d-gaze'%(i+1)] = df['TC_gaze'].values
            combined_data.loc[df.index,'tv%d-exponly'%(i+1)] = df['TC_exposure_only'].values
        
    return combined_data

def reduce_summary(summ): # condenses tv off and tv on data to one
    tvon = True
    cols = ['miss_tvon','gaze_tvon','exp_tvon']
    cols_sel = cols[:]
    
    if np.isnan(summ['gaze_tvon'].values[0]):
        tvon = False
        cols = ['miss_tvoff','gaze_tvoff','exp_tvoff']
        cols_sel = ['miss_time','gaze_epc','exp_epc']
    
    reduced_summ = summ[cols_sel]
    return reduced_summ

