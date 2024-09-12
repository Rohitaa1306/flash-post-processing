import os
import pandas as pd
from datetime import timedelta

path_mobile = 'C:\\Users\\u255769\\flash-post-processing\\mobile_data\\processed_data'
mobile_details = 'C:\\Users\\u255769\\flash-post-processing\\mobile_data\\study4_mobile_details.csv'
num_days = 3

df = pd.read_csv(mobile_details, delimiter=',')
df = df[df['mobile_count'] > 0]

android_compliance = []

def process_data_for_device(row):
    ppt = row['ppt_id']
    start_date = row['start_date']
    type_ = row['mobile_type']
    
    file_paths = {
        'Android': ('%s/%s_chronicle_android.csv', '%s/%s_android_final.csv'),
        'iPhone': ('%s/%s_chronicle_ios.csv', '%s/%s_iphone_final.csv'),
        'iPad': ('%s/%s_ipad_data.csv', '%s/%s_ipad_final.csv')
    }

    if type_ not in file_paths:
        return

    csv_path, save_path = file_paths[type_]
    csv_path = csv_path % (path_mobile, str(ppt))
    save_path = save_path % (path_mobile, str(ppt))

    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        return

    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(row)
    
    m_df = pd.read_csv(csv_path, delimiter=',')
    if type_ in ['iPhone', 'iPad']:
        m_df['event_timestamp'] = pd.to_datetime(m_df['date'] + ' ' + m_df['start_timestamp']).dt.tz_localize(None)
    else:
        m_df['event_timestamp'] = pd.to_datetime(m_df['event_timestamp']).dt.tz_localize(None)

    m_df.sort_values('event_timestamp', inplace=True)
    m_df.set_index('event_timestamp', inplace=True)

    start_dts = pd.to_datetime(start_date)
    end_dts = start_dts + timedelta(days=num_days-1, hours=23, minutes=59, seconds=59)
    m_df = m_df[start_dts:end_dts]

    if 'index' in m_df.columns:
        m_df.drop(columns=['index'], inplace=True)
    
    m_df['username'] = m_df['username'].astype(str).replace('nan', 'None')
    m_df_without_other_use = m_df[m_df['username'] != 'Other']
    m_df_without_other_use.to_csv(save_path, sep=',', index=False)
    
    if type_ == 'Android':
        return calculate_compliance(m_df, start_date)
    
    return None

def calculate_compliance(m_df, start_date):
    compliance_data = []
    
    for day_ in range(num_days):
        start_dts = pd.to_datetime(start_date) + timedelta(days=day_, hours=0, minutes=0, seconds=0)
        end_dts = start_dts + timedelta(hours=23, minutes=59, seconds=59)
        
        m_df_day = m_df[start_dts:end_dts]
        m_df1 = m_df_day[m_df_day['username'].str.lower() == 'target child']
        m_df2 = m_df_day[m_df_day['username'].str.lower() == 'none']
        m_df3 = m_df_day[m_df_day['username'].str.lower() == 'other']
        
        def calculate_duration(df):
            start_ts = pd.to_datetime(df['date'] + ' ' + df['start_timestamp'])
            stop_ts = pd.to_datetime(df['date'] + ' ' + df['stop_timestamp'])
            duration = (stop_ts - start_ts).dt.total_seconds().clip(lower=0).sum()
            return duration
        
        tc_duration = calculate_duration(m_df1)
        un_duration = calculate_duration(m_df2)
        ot_duration = calculate_duration(m_df3)
        
        known_use = tc_duration + ot_duration
        total_use = known_use + un_duration
        compliance = 100 * (known_use / total_use) if total_use > 0 else 0
        
        print(f'Day-{day_+1:02d}, Compliance: {compliance:.4f}, TC use (mins): {tc_duration/60:.2f}, Un use (mins): {un_duration/60:.2f}')
        compliance_data.append(compliance)
    
    return compliance_data

for idx in df.index:
    row = df.loc[idx]
    compliance_data = process_data_for_device(row)
    if compliance_data is not None:
        android_compliance.append([row['ppt_id']] + compliance_data)

df_compliance = pd.DataFrame(android_compliance)
df_compliance.columns = ['FamID'] + [f'Day-{i+1:02d}' for i in range(num_days)]
df_compliance.index.name = 'Index'
df_compliance.to_csv('android_compliance.csv', sep=',', index=True, float_format='%.4f')
