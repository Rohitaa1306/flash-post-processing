import os 
import sys

import numpy as np 
import pandas as pd

from datetime import datetime, timedelta

from utils import *

def pd_df_read(file_path):
    column_names = ['Date', 'TimeStamp', 'FrameNum', 'NumFaces', 'TCPresent', 'Phi', 'Theta', 'Sigma', 'Rot.', 'Top', 'Left', 'Bottom', 'Right', 'Tag']
    df = pd.read_csv(file_path, delimiter=' ', names=column_names)

    condition = df['Tag'].isna() # == float('nan')
    tmp = df.loc[condition, ['Right']] #.copy(deep=True)
    df.loc[condition, ['Right']] = float('nan')
    df.loc[condition, ['Tag']] = tmp['Right']
    
#pd_df_read('./tmp.txt')
'''
# Sample DataFrame with a datetime index
index = pd.date_range(start='2023-01-01 00:00:00.123456', periods=5, freq='S')
data = {'Value': [1, 2, 3, 4, 5]}
df = pd.DataFrame(data, index=index)

print(df)

# Drop microseconds from the datetime index
df.index = df.index.floor('S')
new_index = df.index + pd.Timedelta(seconds=1)

# Display the DataFrame with the modified datetime index
print(df['Value'].sum())
'''

#start_date = '2023-11-10'
#num_days = 10
#dates_range = pd.date_range(start=start_date, periods=num_days, freq='D')
#summary_df = pd.DataFrame(index=dates_range, columns=['miss_time', 'gaze', 'exp', 'gaze_e', 'exp_epc'])
#summary_df.index.name = 'Date'
#summary_df.loc[pd.to_datetime('2023-11-10'),'miss']=10.0

print(summary_df)
