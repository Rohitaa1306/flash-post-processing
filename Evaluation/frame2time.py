import pandas as pd
from datetime import datetime, timedelta

start_time = "2023-09-25 12:54:08"
fps = 30
file_path = 'C:\\Users\\u255769\\flash-post-processing\\Evaluation\\606_tc_gaze_epoch_5s_30fps.txt'

data = pd.read_csv(file_path, sep=' ', header=None, usecols=[5, 10], names=['Frame', 'Category'])
data['Category'] = data['Category'].map({'Gaze': 1, 'No-Gaze': 0, 'Uncertain': -1, 'Out-of-Frame': 0})
data['Timestamp'] = pd.to_datetime(start_time) + pd.to_timedelta(data['Frame'] // fps, unit='s')
data = data.drop_duplicates(subset='Timestamp')
data = data.set_index('Timestamp')

new_time_range = pd.date_range(start=start_time, end=data.index.max(), freq='S')
result = data.reindex(new_time_range, method='ffill').reset_index()

result.to_csv('C:\\Users\\u255769\\flash-post-processing\\Evaluation\\output_vatic.csv')

