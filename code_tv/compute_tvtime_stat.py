import os 
import sys

import numpy as np 
import pandas as pd
import yaml


path = '/home/akv/FLASH_PO1/tech_data_post_processed/output'
#path = '/home/akv/FLASH_PO1/flash-tv-post-processing/data/study4_output'
num_days = 10

ppts = os.listdir(path)
ppts.sort()

#ppts_int = [int(p) for p in ppts]

cols = ['start_date','tv_count'] + ['Day-%02d'%(i+1) for i in range(num_days)]
print(cols)
tv_stat = pd.DataFrame(index=ppts, columns=cols)
tv_stat.index.name = 'ppt_id'

print(tv_stat)

gaze_arr = []
exps_arr = []
miss_arr = []
for ppt in ppts:
    
    # read yaml
    with open('%s/%s/%s_tv_info.yaml'%(path,ppt,ppt)) as stream:
        tv_info = yaml.safe_load(stream)    
        
    # get tv count
    tv_count = tv_info['tv_count']
    start_date = tv_info['start_date']
    
    excluded = 0
    gaze = np.zeros((num_days))
    exps = np.zeros((num_days))
    miss = np.zeros((num_days))
    
    for i in range(tv_count):
        i = i+1
        
        #get exclusion count
        ex = tv_info['tv%s'%i]['exclude']
        excluded += int(ex)
        if ex:
            continue
        
        #get daywise summ across tvs
        fpath = '%s/%s/tv%d_summary.csv'%(path,ppt,i)
        df = pd.read_csv(fpath, delimiter=',')
        df.set_index('Date', inplace=True)
        if 'gaze_tvon' not in df.columns:
            gz_val = df['gaze_epc'].values
            xp_val = df['exp_epc'].values
            ms_val = df['miss_time'].values 
            #miss_time
        else:
            gz_val = df['gaze_tvon'].values
            xp_val = df['exp_tvon'].values
            ms_val = df['miss_tvon'].values 
        
        #print(df)
        gaze += gz_val
        exps += xp_val
        miss += ms_val

    tv_count = tv_count - excluded
    gaze_arr.append(gaze)
    exps_arr.append(exps)
    miss_arr.append(miss)
    
    tv_stat.loc[ppt,'tv_count'] = tv_count
    tv_stat.loc[ppt,'start_date'] = start_date

gaze_arr = np.array(gaze_arr)
exps_arr = np.array(exps_arr)    
miss_arr = np.array(miss_arr)    

tv_stat_gaze = tv_stat.copy()
tv_stat_exps = tv_stat.copy()
tv_stat_miss = tv_stat.copy()
print(tv_stat)


for i in range(num_days):
    i=i+1
    
    tv_stat_gaze['Day-%02d'%i] = gaze_arr[:,i-1]
    tv_stat_exps['Day-%02d'%i] = exps_arr[:,i-1]
    tv_stat_miss['Day-%02d'%i] = miss_arr[:,i-1]
    
print(tv_stat_gaze)
print(tv_stat_exps)

tv_stat_gaze.to_csv('./tv_gaze.csv', sep=',') 
tv_stat_exps.to_csv('./tv_exp_only.csv', sep=',') 
tv_stat_miss.to_csv('./tv_miss.csv', sep=',') 

