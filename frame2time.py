import pandas as pd
from datetime import datetime, timedelta

start_time, fps = "12:54:08", 30
file_path = 'C:\\Users\\u255769\\Downloads\\Evaluation\\606_tc_gaze_epoch_5s_30fps.txt'

data = pd.read_csv(file_path, sep=' ', header=None, usecols=[5, 10], names=['Frame', 'Category'])
data['Category'] = data['Category'].map({'Gaze': 1, 'No-Gaze': 0, 'Uncertain': 0, 'Out-of-Frame': 0})
data['Timestamp'] = data['Frame'].apply(lambda x: (datetime.strptime(start_time, "%H:%M:%S") + timedelta(seconds=x/fps)))
data = data.drop_duplicates(subset='Timestamp')

gaze_data = data[data['Category'] == 1].sort_values(by='Timestamp')
gaze_data['Duration'] = gaze_data['Timestamp'].diff().dt.total_seconds()

total_time_seconds = gaze_data['Duration'].sum()

print(f"Total time in seconds where Category is 1: {total_time_seconds}")