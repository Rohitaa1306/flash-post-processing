import os 
import sys

import numpy as np 
import pandas as pd

from datetime import datetime, timedelta

datetime_format = "%Y-%m-%d %H:%M:%S.%f"    # Datetime format corresponding to the datetime string

def create_dict(keys):
    d = {}
    for key in keys:
        d[key] = {'time':None,'ts_list':[]}
    return d

def convert_to_datetime(date_string, date_format):
    return datetime.strptime(date_string, date_format)
    
def read_first_and_last_line(filename):
    #print(filename)
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        
        last_line = None
        prev_line = None
        for last_line in file:
            #print(len(last_line))
            if len(last_line) > 150:
                last_line = prev_line
                pass
            else:
                prev_line = last_line
                pass  # Reading through the file to reach the last line
            
        if last_line is None:
            last_line = first_line
        
            
    first_date = ' '.join(first_line.split(' ')[:2])
    last_date = ' '.join(last_line.strip().split(' ')[:2])
    return [convert_to_datetime(first_date, datetime_format), convert_to_datetime(last_date, datetime_format)]

def find_files_with_date(data_dict, target_date):
    
    matching_files = []
    for filename, timestamps in data_dict.items():
        start_datetime, stop_datetime = timestamps
        #start_datetime = datetime.strptime(start_timestamp, "%Y-%m-%d %H:%M:%S")
        #stop_datetime = datetime.strptime(stop_timestamp, "%Y-%m-%d %H:%M:%S")
        if start_datetime.date() <= target_date.date() <= stop_datetime.date():
            matching_files.append(filename)
    return matching_files
    
def sort_files(txt_path, start_date, num_days=10):
    file_ls = os.listdir(txt_path)
    file_ls.sort()
    file_ls = [file_ for file_ in file_ls if file_.endswith('.txt')]
    
    #regs = [line for line in file_ls if line.endswith('_reg.txt')]
    #rots = [line for line in file_ls if line.endswith('_rot.txt')]
    txts = [line for line in file_ls if (not line.endswith('_rot.txt') and not line.endswith('_reg.txt'))]
    
    txt_dates = {}
    for each_file in txts:
        txt_dates[each_file] = read_first_and_last_line(os.path.join(txt_path, each_file))
    
    #num_days = 10
    target_date = datetime.strptime(start_date, "%Y-%m-%d")

    date_match = {}
    for day in range(num_days):
        incremented_date = target_date + timedelta(days=day)
        matching_files = find_files_with_date(txt_dates, incremented_date)
        
        date_match[incremented_date.date()] = matching_files
    
    return date_match, txt_dates 
    
def pd_df_read(file_path):
    #No-face-detected, Face-detected, Gaze-no-det, Gaze-det
    
    column_names = ['date', 'timeStamp', 'frameNum', 'numFaces', 'tcPresent', 'phi', 'theta', 'sigma', 'rot.', 'top', 'left', 'bottom', 'right', 'tag']
    df = pd.read_csv(file_path, delimiter=' ', names=column_names)
    
    condition = df['tag'].isna() # == float('nan')
    tmp = df.loc[condition, ['right']] #.copy(deep=True)
    df.loc[condition, ['right']] = float('nan')
    df.loc[condition, ['tag']] = tmp['right']
    
    df['right'] = df['right'].astype(float)
    df['tag'] = df['tag'].astype(str)
    df['dateTimeStamp'] = df['date'] +' '+ df['timeStamp']
    df.drop(['date', 'timeStamp'], axis=1, inplace=True)
    
    column_order = ['dateTimeStamp', 'frameNum', 'numFaces', 'tcPresent', 'phi', 'theta', 'sigma', 'rot.', 'top', 'left', 'bottom', 'right', 'tag']
    df = df[column_order]
    #print(df[df['tag']=='Gaze-det'])
    return df

def get_df_date(txt_path, file_ls, date_key):
    
    df_list = []
    for filename in file_ls:
        # ['dateTimeStamp', 'frameNum', 'numFaces', 'tcPresent', 'phi', 'theta', 'sigma', 'rot.', 'top', 'left', 'bottom', 'right', 'tag']
        filepath = os.path.join(txt_path, filename)
        if os.path.exists(filepath):
            txt_ = pd_df_read(filepath)
            txt_['dateTimeStamp'] = pd.to_datetime(txt_['dateTimeStamp'])
            
            filtered_df = txt_[txt_['dateTimeStamp'].dt.date == date_key]
            df_list.append(filtered_df)
        else:
            df = pd.DataFrame(columns=['dateTimeStamp', 'frameNum', 'numFaces', 'tcPresent', 'phi', 'theta', 'sigma', 'rot.', 'top', 'left', 'bottom', 'right', 'tag'])
            df['dateTimeStamp'] = pd.to_datetime(df['dateTimeStamp'])
            df_list.append(df)
            
    return df_list

day_start = '00:00:00.000000'
day_end = '23:59:59.999998'

def get_miss_time(txt_path, date_key, file_ls):
    
    file_count = 0
    last_ts = None
    miss_time = 0
    ts_list = []
    df_list = []

    for filename in file_ls:
        #print(filename)
        # ['dateTimeStamp', 'frameNum', 'numFaces', 'tcPresent', 'phi', 'theta', 'sigma', 'rot.', 'top', 'left', 'bottom', 'right', 'tag']
        txt_ = pd_df_read(os.path.join(txt_path, filename))
        #print(txt_['dateTimeStamp'])
        txt_['dateTimeStamp'] = pd.to_datetime(txt_['dateTimeStamp'])
        
        filtered_df = txt_[txt_['dateTimeStamp'].dt.date == date_key]
        df_list.append(filtered_df)
        
        first_ts = filtered_df.iloc[0]['dateTimeStamp']

        if file_count == 0:
            day_start_time = pd.to_datetime(str(date_key) + ' ' + day_start) #, datetime_format)
            delta = first_ts - day_start_time
            delta = delta.total_seconds()
            

            miss_time += delta
            
            if delta > 20:
                ts_list.append([first_ts, day_start_time])
                
            
        if last_ts is not None:
            delta = first_ts - last_ts 
            delta = delta.total_seconds()

            miss_time += delta
            ts_list.append([last_ts, first_ts])
                        
        last_ts = filtered_df.iloc[-1]['dateTimeStamp']

        file_count += 1 
        
    #day_end_time = convert_to_datetime(str(date_key) + ' ' + day_end, datetime_format)
    if last_ts is not None:
        day_end_time = pd.to_datetime(str(date_key) + ' ' + day_end)
        delta = day_end_time - last_ts
        delta = delta.total_seconds()
        miss_time += delta
    else:
        delta = 0

    if delta > 20:
        ts_list.append([last_ts, day_end_time])
    
    #print(ts_list)
    return miss_time, ts_list, df_list
    
def epoch_vote(arr):
    #print(arr.shape) 5,
    #vals = [0,1,2,3]
    votes = [(arr==0).sum(), (arr==1).sum(), (arr==2).sum(), (arr==3).sum()]
    idx = np.argmax(votes)
    val = votes[idx]
    
    if val >= 3:
        return idx
    else:
        return 0 
    
def condense_epc(tv_time, tv_exp_only):
    assert tv_time.shape == tv_exp_only.shape
    assert tv_time.size == 86400
    
    tv_time = tv_time.reshape(-1,5)
    tv_exp_only = tv_exp_only.reshape(-1,5)
    
    tv_time_epc = np.apply_along_axis(epoch_vote, axis=1, arr=tv_time)
    tv_exp_only_epc = np.apply_along_axis(epoch_vote, axis=1, arr=tv_exp_only)
    return tv_time_epc, tv_exp_only_epc
    
    
    
