import pandas as pd
from sklearn.metrics import confusion_matrix

file_path = 'C:\\Users\\u255769\\Downloads\\Evaluation\\606_flash_tvdata.txt'
df_flash_pred = pd.read_csv(file_path)

file_path_1 = 'C:\\Users\\u255769\\Downloads\\Evaluation\\output.csv'
df_my_pred = pd.read_csv(file_path_1)

df_my_pred['dateTimeStamp'] = pd.to_datetime(df_my_pred['dateTimeStamp'])
df_flash_pred['timestamp'] = pd.to_datetime(df_flash_pred['timestamp']).dt.floor('S')  # Floor to second

merged_data = pd.merge(df_my_pred, df_flash_pred, left_on='dateTimeStamp', right_on='timestamp', how='inner')

cm = confusion_matrix(merged_data['TC_gaze'], merged_data['flash_prediction'])

TN, FP, FN, TP = cm.ravel()

sensitivity = TP / (TP + FN) 
specificity = TN / (TN + FP)  
accuracy = (TP + TN) / (TP + TN + FP + FN)

print("Confusion Matrix:\n", cm)
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Accuracy: {accuracy:.2f}")
