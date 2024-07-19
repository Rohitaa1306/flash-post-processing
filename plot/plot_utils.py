import os 
import sys

import numpy as np 
import pandas as pd
from datetime import timedelta

def epoch_vote(arr, values, thresh):
    #print(arr.shape) 5,
    #vals = [0,1,2,3]
    #votes = [(arr==0).sum(), (arr==1).sum(), (arr==2).sum()]
    votes = [(arr==val).sum() for val in values]
    idx = np.argmax(votes)
    val = votes[idx]
    
    if val >= thresh: #6*2
        return idx
    else:
        return 0 
        
def condense_arr(data_arr, length, threshold, vals, return_same=True):
    
    #print('asdf1')
    #print((data_arr==1).sum(), (data_arr==2).sum())
    arr_epc = np.apply_along_axis(epoch_vote, axis=1, arr=data_arr.reshape(-1,length), values=vals, thresh=threshold)
    if return_same:
        arr_epc = np.stack([arr_epc]*length,axis=1)
        arr_epc = arr_epc.reshape(-1)
        assert data_arr.size==arr_epc.size
    #print('asdf2')
    #print((arr_epc==1).sum(), (arr_epc==2).sum())
    return arr_epc
    
def plot_tv_col(df, var):
    cols_filtered = [col for col in df.columns if var in col]
    
    gzf, exf = [], []
    missf, downf = [], []
    if 'tv' in var and len(cols_filtered)>0:
        gz = df[var+'-gaze'].values
        ex = df[var+'-exponly'].values
        mix = 1*(gz==1) * 1*(ex==1)
        assert mix.sum()==0
        
        tv_time = ((gz==1).sum() + (ex==1).sum())/12
        miss_time = (ex==2).sum()/12
        down_time = (ex==3).sum()/12
        
        print(var)
        print('Gaze \t: %.2f'%((gz==1).sum()/12))
        print('Expo \t: %.2f'%((ex==1).sum()/12))
        print('Miss \t: %.2f'%((ex==2).sum()/12))
        print('Down \t: %.2f'%((ex==3).sum()/12))
        
        mixed = gz
        mixed[ex==1] = 4
        mixed = condense_arr(mixed, length=12*3, threshold=6*2, vals=[0,1,2,3,4])
        
        gz = mixed.copy()
        ex = mixed.copy()
        gz[gz==4] = 0
        
        ex[ex==1] = 0
        ex[ex==4] = 1
        mix = 1*(gz==1) * 1*(ex==1)
        assert mix.sum()==0
        
        if tv_time >= 10.0:
            gzf = np.where(gz==1)[0]
            exf = np.where(ex==1)[0]
        if miss_time >= 10.0:
            missf = np.where(gz==2)[0]
        if down_time >= 10.0:
            downf = np.where(gz==3)[0]
            
    return [gzf, exf, missf, downf]

def plot_activity(day_df):
    #Non-Wear Time/Wear Time: 0 is non-wear time, 1 is wear time
    #Hand-Scored Wake/Sleep: 0 is wake, 1 is sleep
    #Sadeh Wake/Sleep: 0 is wake, 1 is sleep
    #PA Chandler 2015 8-12 yo S-L-MVPA VM: 0 is Sedentary, 1 is Light, 2 is MVPA
    
    wear_data = day_df['WearTime'].values
    hand_sleep = day_df['HndScrSlp'].values
    sadeh_sleep = day_df['SadehSlp'].values
    chand_mvpa = day_df['ChandlerMVPA'].values
    chand_mvpa = condense_arr(chand_mvpa, length=12*3, threshold=6*2, vals=[0,1,2])
    
    wear_filt = 1*(wear_data==1)
    hand_sleep = hand_sleep*wear_filt
    #chand_mvpa = chand_mvpa*wear_filt
    
    wear_f = np.where(wear_data==1.0)[0]
    nwear_f = np.where(wear_data==0.0)[0]
    wear_miss = np.where(np.isnan(wear_data)==True)[0]
    
    hand_f = np.where(hand_sleep==1.0)[0]
    sadeh_f = np.where(sadeh_sleep==1.0)[0]
    
    chand_0 = np.where(chand_mvpa==0)[0]
    chand_1 = np.where(chand_mvpa==1)[0]
    chand_2 = np.where(chand_mvpa==2)[0]
    #print(chand_mvpa.shape)
    #print(chand_0.shape, chand_1.shape, chand_2.shape)
    
    print('Activity:')
    print('Wear \t: %.2f'%((wear_data==1.0).sum()/720))
    print('Wear_m \t: %.2f'%((np.isnan(wear_data)==True).sum()/720))
    print('Hand \t: %.2f'%((hand_sleep==1.0).sum()/720))
    #print('Sadeh \t: %.2f'%((sadeh_sleep==1.0).sum()/720))
    
    act_miss = [np.concatenate((nwear_f,wear_miss))]
    act_data = [wear_f, hand_f, chand_0, chand_1, chand_2]
    return act_data, act_miss

def get_mobile_data(path_mobile, num_days, ppt, ppt_mobile):
    assert len(ppt_mobile)<2
    ppt_data = ppt_mobile.iloc[0]
    print(ppt_data)
    start_date = ppt_data['start_date']
    count = ppt_data['mobile_count']
    type_ = ppt_data['mobile_type']
    type_ = type_.split(';')
    
    start_dts = pd.to_datetime(start_date) #+ timedelta(days=day)
    end_dts = start_dts + timedelta(days=num_days-1,hours=23,minutes=59,seconds=59)
    m_df_5sec = pd.DataFrame(index=pd.date_range(start=start_dts, end=end_dts, freq='5S'))
    
    #print('Mobile data : ')    
    dev_stats = []
    user_5sec_devs = None
    for dev in range(count):
        mpath = '%s/%s_%s_final.csv'%(path_mobile,ppt,type_[dev].lower())
        
        if not os.path.exists(mpath):
            dev_stats.append(False)
            continue
        else:
            dev_stats.append(True)
        
        print('Dev-%02d : %s'%(dev+1, type_[dev]))    
        
        m_df = pd.read_csv(mpath, delimiter=',')
        m_df['event_timestamp'] = pd.to_datetime(m_df['event_timestamp']).dt.tz_localize(None)
        m_df.set_index('event_timestamp', inplace=True)
        
        
        m_df = m_df[start_dts:end_dts]
        m_df.username = m_df.username.astype(str)
        m_df['user_val'] = m_df['username'].apply(lambda x : 1 if x.lower()=="target child" else 2)
        #print(m_df[m_df['username']!='Target child'])
        
        m_df_sec = pd.DataFrame(index=pd.date_range(start=start_dts, end=end_dts, freq='S'))
        m_df_sec['user'] = 0
        #print(m_df_sec)
        
        for idx, row in m_df.iterrows():
            strt = pd.to_datetime(row['date'] + ' ' + row['start_timestamp'])
            end = pd.to_datetime(row['date'] + ' ' + row['stop_timestamp'])
            m_df_sec[strt:end] = row['user_val']
            
        user_sec = m_df_sec['user'].values       
        
        user_5sec = condense_arr(user_sec, length=5, threshold=3, vals=[0,1,2],return_same=False)
        if user_5sec_devs is None:
            user_5sec_devs = user_5sec.copy()
        else:
            #user_5sec_devs = merge_mobile_use(user_5sec_devs, user_5sec)
            comb_idx = 1*(user_5sec>0) * 1*(user_5sec_devs>0)
            comb_idx = comb_idx==1
            
            user_5sec_devs[user_5sec_devs==0] = user_5sec[user_5sec_devs==0]
            user_5sec_devs[comb_idx] = user_5sec[comb_idx]
            
            #comb == 0; replace with new one values
            #comb > 0; replace with new ones nonzero values

        
        print('Child use: %.2f'%((user_sec==1).sum()/60.0))
        print('None  use: %.2f'%((user_sec==2).sum()/60.0))
    
    print('Total:')
    print('Child use: %.2f'%((user_5sec_devs==1).sum()/12.0))
    print('None  use: %.2f'%((user_5sec_devs==2).sum()/12.0))
        
    m_df_5sec['user'] = user_5sec_devs
        
    return m_df_5sec, dev_stats
    

