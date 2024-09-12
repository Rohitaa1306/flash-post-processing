import pandas as pd

from datetime import timedelta

def convert_secs(time):
    h,m,s = time.split(':')
    h = int(h)
    m = int(m)
    s = int(s[:2])
    
    
    n = h*3600 + m*60 + s
    #print(time, h,m,s, n)
    return n

ppt = '608'
start_date = '2023-09-20'

d = pd.read_excel('C:\\Users\\u255769\\flash-post-processing\\mobile_data\\raw_data\\Screen Time FLASH '+ppt+'.xlsx')

save_name = ppt+'_ipad_data.csv'

d['Date'] = pd.to_datetime(d['Date'])
num_days = 3

#day = 0
dates = []
ppts = []
start_ts = []
stop_ts = []
epochs = []
apps = []
user = []

for day in range(num_days):
    start_dts = pd.to_datetime(start_date) #+ timedelta(days=day)
    start_dts = start_dts + timedelta(days=day)
    #print(start_dts)        

    d_ = d[d['Date']==start_dts]

    print(d_)
    d_ = d_.drop(d_[d_['Title'] == 'Daily App by App Total'].index)
    d_ = d_.drop(d_[d_['Title'] == 'Daily Displayed Total'].index)
    #d_ = d_.iloc[2:]
    print(d_)

    if d_.shape[0]>=1:
        
        for hr in range(24):
            #print(d_[hr])
            assert d_[hr].sum() <= 60
            epoch_window = '%02d:%02d:%02d'%(hr, 0, 0)
            prev_start = None
            for idx, dur_m in enumerate(d_[hr]):
                if dur_m > 0:
                    #print(hr, dur_m)
                    start_m = 0 if prev_start is None else start_m + prev_start
                    
                    
                    time_start = '%02d:%02d:%02d'%(hr, start_m, 0)
                    time_end = '%02d:%02d:%02d'%(hr, start_m+dur_m, 0)
                    assert time_end > time_start

                    prev_start = dur_m
                    app_name = d_.iloc[idx,d_.columns.get_loc('Title')]
                    print(start_dts.date(), time_start, time_end, app_name)
                    
                    dates.append(start_dts.date())
                    start_ts.append(time_start)
                    stop_ts.append(time_end)
                    epochs.append(epoch_window)
                    apps.append(app_name)
                    user.append('Target child')
                    ppts.append(ppt)

#date,start_timestamp,stop_timestamp                
data = {}
data['participant_id'] = ppts
data['date'] = dates
data['epoch_window'] = epochs
data['start_timestamp'] = start_ts
data['stop_timestamp'] = stop_ts
data['username'] = user
data['application_label'] = apps

df = pd.DataFrame(data)
print(df)
df.index.name = 'index'
df.to_csv(save_name, sep=',')
