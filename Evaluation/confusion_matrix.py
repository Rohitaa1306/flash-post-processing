import pandas as pd
from sklearn.metrics import confusion_matrix

file_path = 'C:\\Users\\u255769\\flash-post-processing\\Evaluation\\606_flash_tvdata.txt'
df_flash_pred = pd.read_csv(file_path)

file_path_1 = 'C:\\Users\\u255769\\flash-post-processing\\Evaluation\\output_1.csv'
df_my_pred = pd.read_csv(file_path_1)

df_my_pred['dateTimeStamp'] = pd.to_datetime(df_my_pred['dateTimeStamp'])
df_flash_pred['timestamp'] = pd.to_datetime(df_flash_pred['timestamp']).dt.floor('S') 

merged_data = pd.merge(df_my_pred, df_flash_pred, left_on='dateTimeStamp', right_on='timestamp', how='inner')
merged_data['Gold_standard'] = merged_data['Gold_standard'].map({1: 1, 0: 0, 2: 0, 5: 0})

cm = confusion_matrix(merged_data['TC_gaze'], merged_data['Gold_standard'])

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

cm = confusion_matrix(merged_data['TC_gaze'], merged_data['flash_prediction'])

TN, FP, FN, TP = cm.ravel()

sensitivity = TP / (TP + FN) 
specificity = TN / (TN + FP)  
accuracy = (TP + TN) / (TP + TN + FP + FN)

true_label = 'TC_gaze'
predicted_label = 'flash_prediction'

print(f"Confusion Matrix: {true_label} vs {predicted_label}")
print(f"                      {predicted_label} = 0   {predicted_label} = 1")
print(f" {true_label} = 0    |        {TN}                    {FP}")
print(f" {true_label} = 1    |        {FN}                    {TP}")
print("\nMetrics:")
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Accuracy: {accuracy:.2f}")
