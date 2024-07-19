import os
import sys

import numpy as np
import pandas as pd
import yaml

from datetime import timedelta

# Define paths
path_mobile = 'flash-post-processing\code_mobile' #Replace with full path name 
mobile_details = 'flash-post-processing\code_mobilestudy4_mobile_details.csv' #Replace with full path name
num_days = 3

# Load data
df = pd.read_csv(mobile_details, delimiter=',')
df = df[df['mobile_count'] > 0]

def process_mobile_data(row, num_days, path_mobile):
    ppt = row['ppt_id']
    start_date = row['start_date']
    type_ = row['mobile_type']

    # Determine csv path based on mobile type
    if type_.lower() == 'android':
        csv_path = f'{path_mobile}/{ppt}_chronicle_android.csv'
        save_path = f'{path_mobile}/{ppt}_android_final.csv'
    elif type_.lower() == 'ios':
        csv_path = f'{path_mobile}/{ppt}_chronicle_ios.csv'
        save_path = f'{path_mobile}/{ppt}_ios_final.csv'
    elif type_.lower() == 'ipad':
        csv_path = f'{path_mobile}/{ppt}_chronicle_ipad.csv'
        save_path = f'{path_mobile}/{ppt}_ipad_final.csv'
    else:
        print(f"Unknown mobile type: {type_}")
        return None

    # Load mobile data
    m_df = pd.read_csv(csv_path, delimiter=',')
    m_df['event_timestamp'] = pd.to_datetime(m_df['event_timestamp']).dt.tz_localize(None)

    m_df.sort_values('event_timestamp', inplace=True)
    m_df.set_index('event_timestamp', inplace=True)

    # Filter data within the given date range
    start_dts = pd.to_datetime(start_date)
    end_dts = start_dts + timedelta(days=num_days-1, hours=23, minutes=59, seconds=59)
    m_df = m_df[start_dts:end_dts]
    if 'index' in m_df.columns:
        m_df.drop(columns=['index'], inplace=True)
    m_df['username'] = m_df['username'].astype(str).apply(lambda x: "None" if x == "nan" else x)

    # Save filtered data
    m_df_without_other_use = m_df[m_df['username'] != 'Other']
    m_df_without_other_use.to_csv(save_path, sep=',')

    return m_df, start_date

def calculate_compliance(m_df, start_date, num_days):
    compliance_data = []
    for day_ in range(num_days):
        start_dts = pd.to_datetime(start_date) + timedelta(days=day_, hours=0, minutes=0, seconds=0)
        end_dts = start_dts + timedelta(hours=23, minutes=59, seconds=59)
        m_df_day = m_df[start_dts:end_dts]

        tc_duration, un_duration, ot_duration = 0, 0, 0

        for username in ['target child', 'none', 'other']:
            duration = 0
            start_ts = pd.to_datetime(m_df_day.loc[m_df_day['username'].str.lower() == username, 'date'] + ' ' + m_df_day.loc[m_df_day['username'].str.lower() == username, 'start_timestamp'])
            stop_ts = pd.to_datetime(m_df_day.loc[m_df_day['username'].str.lower() == username, 'date'] + ' ' + m_df_day.loc[m_df_day['username'].str.lower() == username, 'stop_timestamp'])
            duration = (stop_ts - start_ts).dt.total_seconds().sum()
            assert duration >= 0

            if username == 'target child':
                tc_duration = duration
            elif username == 'none':
                un_duration = duration
            elif username == 'other':
                ot_duration = duration

        known_use = tc_duration + ot_duration
        total_use = known_use + un_duration
        if total_use > 0:
            compliance = 100 * (known_use / total_use)
        else:
            compliance = 0
        print(f'Day-{day_+1:02d}, Compliance: {compliance:.1f}, TC use (mins): {tc_duration / 60:.1f}, Un use (mins): {un_duration / 60:.1f}')
        compliance_data.append(compliance)

    return compliance_data

compliance_data_all = []

for idx, row in df.iterrows():
    result = process_mobile_data(row, num_days, path_mobile)
    if result:
        m_df, start_date = result
        compliance_data = calculate_compliance(m_df, start_date, num_days)
        compliance_data_all.append([row['ppt_id'], row['mobile_type']] + compliance_data)

# Save all compliance data
columns = ['FamID', 'MobileType'] + [f'Day-{i+1:02d}' for i in range(num_days)]
df_compliance_all = pd.DataFrame(compliance_data_all, columns=columns)
df_compliance_all.index.name = 'Index'
print(df_compliance_all)
df_compliance_all.to_csv('compliance_all_types.csv', sep=',')
