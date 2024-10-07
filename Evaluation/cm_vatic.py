import pandas as pd
from sklearn.metrics import confusion_matrix

file_path = 'C:\\Users\\u255769\\flash-post-processing\\Evaluation\\output_5.csv'
df = pd.read_csv(file_path)

file_path_1 = 'C:\\Users\\u255769\\flash-post-processing\\Evaluation\\output_vatic.csv'
df_vatic = pd.read_csv(file_path_1)

df_vatic['index'] = pd.to_datetime(df_vatic['index'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

end_time = pd.to_datetime('14:01:52').time()
start_time = pd.to_datetime('12:54:08').time()
df_vatic = df_vatic[(df_vatic['index'].dt.time >= start_time) & (df_vatic['index'].dt.time <= end_time)]

merged_data = pd.merge(df, df_vatic, left_on='timestamp', right_on='index', how='inner')
merged_data = merged_data.dropna(subset=['TC_gaze', 'Category'])

print(merged_data.shape)
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
total_time_gt = gaze_1['timestamp'].diff().sum()
total_time_pred = gaze_pred_1['timestamp'].diff().sum()
total_time_gt_mins = total_time_gt.total_seconds() / 60
total_time_pred_mins = total_time_pred.total_seconds() / 60

print(f"Total time where TC_gaze is 1: {total_time_pred_mins} minutes")
print(f"Total time where Category is 1: {total_time_gt_mins} minutes")
