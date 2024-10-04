import pandas as pd
import numpy as np 

from pandas import read_csv
from matplotlib import pyplot as plt

from pandas import Grouper
from pandas import DataFrame

from datetime import timedelta

# Function to create the DateTimeIndex and power values DataFrame
def create_datetime_power_df(start_date, end_date):
    dt_index = pd.date_range(start=start_date, end=end_date, freq='5S')
    power_values = np.random.randint(253, 255, size=len(dt_index))
    power_df = pd.DataFrame({'power': power_values}, index=dt_index)
    power_df.index.name='datetime1'
    return power_df
    
#file_name = '/home/akv/FLASH_PO1/techData/1024/P1-1024006_data/P1-1024006_tv_power_5s_.csv'
#file_name = '/home/akv/FLASH_PO1/techData/1036/P1-1036007_data/P1-1036007_tv_power_5s_.csv'
#file_name = '/home/akv/FLASH_PO1/data/596/596003_data/596003_tv_power_5s.csv'
#file_name = '/home/akv/FLASH_PO1/data/600/600003_data/600003_tv_power_5s.csv'

#file_name = '/media/akv/FLASHSYS/Anil_data/data/608/608014_data/608014_tv_power_5s.csv'
def read_power_pd(file_path):
    series = read_csv(file_path, delimiter=';',header=0, index_col=None, parse_dates=True,on_bad_lines='skip')

    series['time'] = series['time'].str.replace('.',':')
    series['time'] = series['time'].str.replace(' ','')
    series['power'] = pd.to_numeric(series['power'], errors='coerce')

    series['datetime'] = series['date'] + ' ' + series['time']
    series['datetime'] = pd.to_datetime(series['datetime'])


    #series['date'] = pd.to_datetime(series['date'])
    series = series.drop(columns=['date','time'])
    series = series.set_index(['datetime'])
    return series 

def filter_series(series, start_date, end_date):
    #ns = series[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    ns = series[series.index<=pd.to_datetime(end_date)]
    ns = ns[ns.index>=pd.to_datetime(start_date)]
    return ns

def interp_series(series, start_date, end_date, visualize=False):
    # Create DataFrame from the datetime index and power values
    df = create_datetime_power_df(start_date, end_date)

    # Merge the DataFrames based on the datetime index (fill in the current df with power values from input_df)
    merged_df = df.merge(series, left_index=True, right_on='datetime', how='left')
    if len(series) <= len(df) and merged_df.index.name!='datetime':
        merged_df.set_index('datetime', inplace=True)

    merged_df.drop(columns=['power_x'], inplace=True)
    merged_df.rename(columns={'power_y': 'power'}, inplace=True)

    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
    unknown_time = merged_df.isna().sum().iloc[0]*5/60.0
    print('Unknown duration (mins): %.2f'%unknown_time)
    
    if not visualize:
        merged_df.fillna(250, inplace=True)
    #print('Unknown values:', merged_df.isna().sum())
    
    return merged_df

if __name__ == "__main__":
    file_path = 'C:\\Users\\u255769\\tv_mobile_data\\tv_data\\data\\P1-1575\\P1-1575010_data\\P1-1575010_data\\P1-1575010_tv_power_5s.csv'
    start_date = '2024-09-29'
    num_days = 5
    end_date = pd.to_datetime(start_date) + timedelta(days=num_days-1)
    
    start_date = start_date+' 00:00:00'
    end_date = str(end_date)+' 23:59:55'
    
    print(file_path)
    series = read_power_pd(file_path)
    ns = filter_series(series, start_date, end_date)
    interp_df = interp_series(ns, start_date, end_date, visualize=True)
    print(interp_df)

    #start_date = '2023-11-10 00:00:00'
    #end_date = '2023-11-19 23:59:55'

    plot_df = interp_df[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    groups = plot_df.groupby(Grouper(freq='D'))
    print(len(groups))

    days = DataFrame()
    for name, group in groups:
        print(name)
        days[name.date()] = group.values.reshape(-1)
        #days[name.date()] = group['power'] 

    days['time'] = group.index
    days = days.set_index(['time'])

    #print(days)
    days.plot(figsize=(13, 4.5), subplots=True, legend=False, ylim=(-2,10))
    plt.savefig(file_path.split('/')[-1][:-4]+'.png')
    #plt.show()
