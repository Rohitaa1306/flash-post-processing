import os
import random
import math
import argparse
import time

import pandas as pd
import numpy as np

from datetime import datetime
import datetime as dt

def convert_secs(time):
    h,m,s = time.split(':')
    h = int(h)
    m = int(m)
    s = int(s[:2])
    
    
    n = h*3600 + m*60 + s
    #print(time, h,m,s, n)
    return n


parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str,
                    default='./test_android.csv', help='path to the raw logs from Chronicle dashboard')
parser.add_argument('--save', type=str,
                    default=None, help='the processed log is stored here if not specified it generates a name based on participant ID')
parser.add_argument('--survey', type=str,
                    default=None, help='survey data to fill-in missed notifications')

parser.add_argument('--date', type=str,
                    default=None, help='please input date in YYYY-MM-DD format to save the data for the specific date')

parser.add_argument('--only-child', type=int,
                    default=0, help='please input whether child is the sole user, 1 for child only, 0 for shared')

args = parser.parse_args()
                    

interaction_types = ['Move to Foreground', 'Move to Background']

fname = args.csv    
save_date = args.date
child = args.only_child

df = pd.read_csv(fname)
df = df[df['interaction_type'].isin(interaction_types)]
ppt_id = df.iloc[1]['participant_id']

if args.save is None:
    save_name = str(ppt_id) + '_chronicle_android.csv'
else:
    save_name = '%s/%s_chronicle_android.csv'%(args.save, ppt_id)

# Sort the DataFrame by "event_timestamp" column to ensure the rows are in chronological order
df.sort_values('event_timestamp', inplace=True)

# Reset the index of the DataFrame
df.reset_index(drop=True, inplace=True)

# Create new columns for start and stop timestamps
df['start_timestamp'] = pd.NaT
df['stop_timestamp'] = pd.NaT

# Iterate over the DataFrame rows until the second-to-last row
for index, row in df.iterrows():
    if index < len(df) - 1:
        next_row = df.iloc[index + 1]
        if row['interaction_type'] == 'Move to Foreground':
            # Check if the next row has the same app_package_name, application_label, and interaction_type as 'Move to Background'
            if next_row['app_package_name'] == row['app_package_name'] and \
                    next_row['application_label'] == row['application_label'] and \
                    next_row['interaction_type'] == 'Move to Background':
                
                #print(df.iloc[index:index+2])
                # Set the start and stop timestamps
                df.at[index, 'start_timestamp'] = pd.to_datetime(row['event_timestamp'])
                df.at[index, 'stop_timestamp'] = pd.to_datetime(next_row['event_timestamp'])

# Remove rows where either the start or stop timestamp is missing
df = df.dropna(subset=['start_timestamp', 'stop_timestamp'])
df['date'] = pd.to_datetime(df['start_timestamp'])
df['date'] = df['date'].dt.date

df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
df['stop_timestamp'] = pd.to_datetime(df['stop_timestamp'])
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

print('Splitting these usage events across the day')
time_wrong_df = df[df['start_timestamp'].dt.date != df['stop_timestamp'].dt.date]
print(time_wrong_df[['date', 'username', 'start_timestamp', 'stop_timestamp']])

for idx in time_wrong_df.index:
    #print(idx)
    #print(time_wrong_df.loc[idx])
    row = df.loc[idx]
    start_ts = row['start_timestamp']
    stop_ts = row['stop_timestamp']
    event_ts = row['event_timestamp']
    
    stop_ts1 = str(start_ts.date()) + ' 23:59:59.999998' + str(event_ts)[-6:]
    start_ts2 = str(stop_ts.date()) + ' 00:00:00.000001' + str(event_ts)[-6:]
    event_ts2 = str(stop_ts.date()) + 'T' + '00:00:00.001' + str(event_ts)[-6:]
    
    stop_ts1 = pd.to_datetime(stop_ts1)
    start_ts2 = pd.to_datetime(start_ts2)
    event_ts2 = pd.to_datetime(event_ts2)
    
    df.at[idx,'stop_timestamp'] = stop_ts1

    row2 = row.copy()
    row2['start_timestamp'] = start_ts2
    row2['event_timestamp'] = event_ts2
    
    df.loc[len(df.index)] = row2
    #print(df.loc[idx])
    #print(row2)


df['start_timestamp'] = df['start_timestamp'].dt.strftime('%H:%M:%S')
df['stop_timestamp'] = df['stop_timestamp'].dt.strftime('%H:%M:%S')

# Reset the index of the DataFrame
df.reset_index(drop=True, inplace=True)

cols = df.columns.tolist()
#cols = ['study_id', 'participant_id', 'date', 'start_timestamp', 'stop_timestamp', 'username', 'application_label', 'app_package_name', 'event_timestamp', 'interaction_type', 'event_type', 'timezone', 'uploaded_at']

filtered_cols = ['study_id', 'participant_id', 'date', 'start_timestamp', 'stop_timestamp', 'username', 'application_label', 'app_package_name', 'event_timestamp', 'timezone']
df = df[filtered_cols]
df.sort_values('event_timestamp', inplace=True)


if not child:
    df.username = df.username.astype(str)
    df.username = df.username.apply(lambda x : "None" if x=="nan" else x)
else:
    df.username = df.username.astype(str)
    df.username = df.username.apply(lambda x : "Target Child" if x=="nan" else x)

#df = df[df.username != 'Other']

if args.survey is not None:
    survey_df = pd.read_csv(args.survey)
    survey_df.sort_values('event_timestamp', inplace=True)
    survey_df.reset_index(drop=True, inplace=True)
    
    for index, row in survey_df.iterrows():
        idxs = df.index[df['event_timestamp']==row['event_timestamp']].tolist()
        if len(idxs) > 1:
            print('Index mismatch.')
            raise
        
        if len(idxs)>0:
            df.at[idxs[0], 'username'] = str(row['users'])

# Print the updated DataFrame
#print(df)
#print(df.iloc[:10])

time_wrong_df = df[df['start_timestamp'] > df['stop_timestamp']]
assert len(time_wrong_df.index) == 0

if save_date is not None:
    new_df = df[df['date']==pd.to_datetime(save_date).date()]
    #if save_name is None:
    file_name = './' + str(ppt_id) + '_' + save_date + '.csv'
    #else:
    #    file_name = save_name
        
    new_df.to_csv(file_name, index=False)
    print('results saved at: ', file_name)
    
    duration_ls = []
    for user_ in ['Target child', 'Other', 'nan']:
        user_df = new_df[new_df['username'] == user_]
        print(user_+' data: \n')
        
        user_durn = 0
        if len(user_df) > 0:
            for index, row in user_df.iterrows():
                strt = row['start_timestamp']
                stop = row['stop_timestamp']
                
                strt_N = convert_secs(strt)
                stop_N = convert_secs(stop)
                if stop_N < strt_N:
                    print('checking', strt, stop, strt_N, stop_N)
                    user_durn = user_durn + (24*3600-strt_N) + (stop_N-0)
                else:
                    user_durn = user_durn + (stop_N - strt_N)
                
            #user_df['start_timestamp'] = pd.to_datetime(user_df['start_timestamp'])
            #user_df['stop_timestamp'] = [time.time() for time in user_df['start_timestamp']]
            
            #user_df['stop_timestamp'] = pd.to_datetime(user_df['stop_timestamp'])
            #user_df['stop_timestamp'] = [time.time() for time in user_df['stop_timestamp']]
            
            #print(user_df)
        #print(user_durn)
        duration_ls.append(user_durn)
    
    duration_ls = np.array(duration_ls)
    print(duration_ls)
    total = duration_ls.sum()
    child = duration_ls[0]
    other = duration_ls[1]
    unknn = duration_ls[2]
    compliance = (total-unknn)/total
    print('compliance: %.1f'%(compliance*100))
    print('child use (hrs): %.1f'%(child/3600))
    print('other use (hrs): %.1f'%(other/3600))
    print('Unknown use (hrs): %.1f'%(unknn/3600))
    
else:
    df.to_csv(save_name, index=False)
    print('results saved at: ', save_name,fmt='%d,%.1f,%.1f,%.1f')
