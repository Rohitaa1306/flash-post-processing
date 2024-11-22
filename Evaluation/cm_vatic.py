import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix

file_path = 'C:\\Users\\u255769\\flash-post-processing\\Evaluation\\output_1.csv'
df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

first_valid_timestamp = df[df['timestamp'].dt.second % 5 == 0].iloc[0]['timestamp']
last_timestamp = df['timestamp'].iloc[-1]
timestamp_range = pd.date_range(start=first_valid_timestamp, end=last_timestamp, freq='5S')

df['timestamp'] = pd.cut(df['timestamp'], bins=timestamp_range, right=False)

def majority(values):
    return values.mode()[0] if not values.mode().empty else values.iloc[0]

df_resampled = df.groupby('timestamp').agg({
    'TC_gaze': majority,
    'TC_exposure_only': majority
}).reset_index()

df_resampled['timestamp'] = df_resampled['timestamp'].apply(lambda x: x.left)

df_resampled.to_csv('output_5.csv', index=False)
print(df_resampled)

file_path_1 = 'C:\\Users\\u255769\\flash-post-processing\\Evaluation\\output_vatic.csv'
df_vatic = pd.read_csv(file_path_1)
df_vatic['index'] = pd.to_datetime(df_vatic['index'])

first_valid_timestamp_vatic = df_vatic[df_vatic['index'].dt.second % 5 == 0].iloc[0]['index']
last_timestamp_vatic = df_vatic['index'].iloc[-1]
timestamp_range_vatic = pd.date_range(start=first_valid_timestamp_vatic, end=last_timestamp_vatic, freq='5S')

df_vatic['index'] = pd.cut(df_vatic['index'], bins=timestamp_range_vatic, right=False)

df_vatic_resampled = df_vatic.groupby('index').agg({
    'Category': majority
}).reset_index()

df_vatic_resampled['index'] = df_vatic_resampled['index'].apply(lambda x: x.left)

df_vatic_resampled.to_csv('output_vatic_5.csv', index=False)
print(df_vatic_resampled)

df_vatic_resampled['index'] = pd.to_datetime(df_vatic_resampled['index'])
df_resampled['timestamp'] = pd.to_datetime(df_resampled['timestamp'])

merged_data = pd.merge(df_resampled, df_vatic_resampled, left_on='timestamp', right_on='index', how='inner')
merged_data = merged_data.dropna(subset=['TC_gaze', 'Category'])

if not merged_data.empty:
    start_time = merged_data['timestamp'].min().time()
    end_time = merged_data['timestamp'].max().time()
    
    df_vatic_resampled = df_vatic_resampled[(df_vatic_resampled['index'].dt.time >= start_time) & (df_vatic_resampled['index'].dt.time <= end_time)]
    
    merged_data = pd.merge(df_resampled, df_vatic_resampled, left_on='timestamp', right_on='index', how='inner')
    merged_data = merged_data.dropna(subset=['TC_gaze', 'Category'])

    cm = confusion_matrix(merged_data['TC_gaze'], merged_data['Category'])

    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()

        sensitivity = TP / (TP + FN) 
        specificity = TN / (TN + FP)  
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        true_label = 'TC_gaze'
        predicted_label = 'Gold_standard'
        
        print(f"Confusion Matrix: {true_label} vs {predicted_label}")
        print(f"                      {predicted_label} = 0   {predicted_label} = 1")
        print(f" {true_label} = 0    |        {TN}                    {FP}")
        print(f" {true_label} = 1    |        {FN}                    {TP}")
        print("\nMetrics:")
        print(f"Sensitivity (Recall): {sensitivity:.2f}")
        print(f"Specificity: {specificity:.2f}")
        print(f"Accuracy: {accuracy:.2f}")

    gaze_pred_1 = merged_data[(merged_data['TC_gaze'] == 1)]
    gaze_1 = merged_data[(merged_data['Category'] == 1)]

    print(f"Total gaze time (FLASH) is : {gaze_1.shape[0] * 5} seconds")
    print(f"Total gaze time (Gold Standard) is : {gaze_pred_1.shape[0] * 5} seconds")
