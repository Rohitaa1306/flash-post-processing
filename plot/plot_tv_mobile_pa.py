import os 
import sys

import numpy as np 
import pandas as pd
from datetime import timedelta

import yaml

import matplotlib
import matplotlib.pyplot as plt

from plot_utils import *

#'darkorange','moccasin'
#'darkkhaki','khaki'
#'dodgerblue','deepskyblue'

# Plot variables
tv_plot_colors = ['darkorange','moccasin','lightgray','darkgray',
                    'darkorange','moccasin','lightgray','darkgray',
                    'darkorange','moccasin','lightgray','darkgray',
                    ]
act_colors = ['dodgerblue','deepskyblue','darkkhaki','khaki','gold']
act_offset = [2.1,3.1,4.1,5.1,6.1]
assert len(act_offset)==len(act_colors)
act_len = [0.78]*len(act_colors)

act_miss_rep = len(act_offset)
act_miss_colors = ['lightgray']*act_miss_rep

plt_ticks = [0,1] + act_offset
plt_labels = ['Mobile','TV','Wear-Time','Sleep Scr', 'Sedentary', 'Light', 'MVPA']


path = '/home/akv/FLASH_PO1/tech_data_post_processed/study4_tv_output'
path_pa = '/home/akv/FLASH_PO1/tech_data_post_processed/study4_pa_data'
path_mobile = '/home/akv/FLASH_PO1/tech_data_post_processed/study4_mobile_output'
mobile_details = './study4_mobile_details.csv'

mobile_details = pd.read_csv(mobile_details, delimiter=',')
num_days=3

ppts = os.listdir(path)
ppts.sort()

#ppts_int = [int(p) for p in ppts]
#ppts[:1]
for ppt in ['578']:

    ###### TV DATA    
    # read yaml
    with open('%s/%s/%s_tv_info.yaml'%(path,ppt,ppt)) as stream:
        tv_info = yaml.safe_load(stream)    
        
    # get tv count
    tv_count = tv_info['tv_count']
    start_date = tv_info['start_date']

    print('Partcpnt-id \t: %s'%ppt)    
    print('Start Date \t: %s'%start_date)    
    print('TV count \t: %d'%tv_count)
    print('Num days \t: %d'%num_days)    
    
    tv_fpath = '%s/%s/%s_tv_viewing_data.csv'%(path, ppt, ppt)
    pa_fpath = '%s/PA Scored FLASH %s.csv'%(path_pa, ppt)
    
    tv_data = pd.read_csv(tv_fpath, delimiter=',')
    tv_data['dateTimeStamp'] = pd.to_datetime(tv_data['dateTimeStamp'])
    tv_data.set_index('dateTimeStamp', inplace=True)
    
    # remove tv columns that are not present (cols with -1 val)
    for col in tv_data.columns:
        val = tv_data[col].values
        if (val==-1).sum() > 100:
            tv_data.drop(columns=[col],inplace=True)
    
    tv_present = any(['tv' in col for col in tv_data.columns])


    ###### MOBILE DATA    
    ppt_mobile = mobile_details[mobile_details['ppt_id'] == int(ppt)]
    no_mobile = ppt_mobile['mobile_count'].values[0] < 1
    assert ppt_mobile['start_date'].values[0]==start_date
    
    if not no_mobile:
        m_df_5sec = get_mobile_data(path_mobile, num_days, ppt, ppt_mobile)

    ###### ACTIVITY DATA    
    pa_data = pd.read_csv(pa_fpath, delimiter=',')
    pa_data = pa_data[['DateTime','Non-Wear Time/Wear Time','Hand-Scored Wake/Sleep','Sadeh Wake/Sleep','PA Chandler 2015 8-12 yo S-L-MVPA VM']]
    pa_data.rename(columns={'Non-Wear Time/Wear Time': 'WearTime', 'Hand-Scored Wake/Sleep': 'HndScrSlp', 'Sadeh Wake/Sleep':'SadehSlp', 'PA Chandler 2015 8-12 yo S-L-MVPA VM':'ChandlerMVPA'}, inplace=True)
    pa_data['DateTime'] = pd.to_datetime(pa_data['DateTime'])    
    pa_data.set_index('DateTime', inplace=True)
    #print(pa_data)
    
    merged_df = tv_data.merge(pa_data, left_index=True, right_on='DateTime', how='left')
    if merged_df.index.name != 'DateTime':
        merged_df.set_index('DateTime', inplace=True)
    merged_df.index.name = 'dateTimeStamp'
    #print(merged_df)
    
    
    for day in range(num_days):
        print('Day : %02d'%(day+1))

        date = pd.to_datetime(start_date) + timedelta(days=day)
        start_dts = date
        end_dts = date + timedelta(hours=23,minutes=59,seconds=55)
        
        day_df = merged_df[start_dts:end_dts]
        
        # mobile data 
        if tv_present: 
            tv1_plot_data = plot_tv_col(day_df, 'tv1') #[gzf, exf, missf, downf]
            tv2_plot_data = plot_tv_col(day_df, 'tv2')
            tv3_plot_data = plot_tv_col(day_df, 'tv3')
            tv_plot_data = tv1_plot_data + tv2_plot_data + tv3_plot_data

        # mobile data 
        if not no_mobile:
            day_mdf = m_df_5sec[start_dts:end_dts]
            m_data = day_mdf['user'].values
            tc_use = np.where(m_data==1)[0]
            un_use = np.where(m_data==2)[0]
            m_plot = [tc_use] + [un_use]
            m_colors = ['olivedrab','yellowgreen']
            m_offset = [0]*2
        
        #activity data
        act_data, act_miss = plot_activity(day_df)
        act_miss = act_miss*act_miss_rep
        
        
        # plot the bar graphs
        title = '%s ScreenTime, %s, %s'%(ppt, date.day_name()[:3], date.date())
        fname = '%s ScreenTime, %s'%(ppt, date.date())
        matplotlib.rcParams['font.size'] = 11.0
        #fig, ax = plt.subplots(1, 1)
        fig = plt.figure(figsize=(15,2))
        
        if tv_present:
            plt.eventplot(tv_plot_data, colors=tv_plot_colors, lineoffsets=[1,1,1,1]*3,
                        linelengths=[0.8]*12)
        else:
            plt.eventplot(range(0,17280,2), colors=['k'], lineoffsets=[1],
                        linelengths=[0.8]*1)
                        
        if no_mobile:
            plt.eventplot(range(0,17280,2), colors=['k'], lineoffsets=[0],
                        linelengths=[0.8]*1)
        else:
            plt.eventplot(m_plot, colors=m_colors, lineoffsets=m_offset,
                        linelengths=[0.8]*1)
                                    
        plt.eventplot(act_data, colors=act_colors, lineoffsets=act_offset,
                    linelengths=act_len)
        plt.eventplot(act_miss, colors=act_miss_colors, lineoffsets=act_offset,
                    linelengths=act_len)
                            
        plt.xticks(ticks=np.arange(0,25*720,2*720), labels=np.arange(0,25,2))
        plt.yticks(ticks=plt_ticks, labels=plt_labels)
        plt.xlim(-100,17380)
        plt.ylim(-1.0,7.0)
        plt.title(title)
        #plt.show()
        plt.savefig('./plots/'+fname+'.png')
        #plt.close()
    


