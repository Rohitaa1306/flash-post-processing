import pandas as pd
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

file_path = "C:\\Users\\u255769\\flash-post-processing\\Evaluation\\output_1.csv"
df = pd.read_csv(file_path)

file_path_1 = "C:\\Users\\u255769\\flash-post-processing\\Evaluation\\output_vatic.csv"
df_vatic = pd.read_csv(file_path_1)

df_vatic["index"] = pd.to_datetime(df_vatic["index"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

intersecting_times = df["timestamp"][df["timestamp"].isin(df_vatic["index"])]

start_time = intersecting_times.min()
stop_time = intersecting_times.max()

start_time = start_time.floor("5S")
time_range = pd.date_range(start=start_time, end=stop_time, freq="5S")

df_filtered = df[df["timestamp"].isin(time_range)]
df_vatic_filtered = df_vatic[df_vatic["index"].isin(time_range)]

merged_data = pd.merge(df_filtered, df_vatic_filtered, left_on="timestamp", right_on="index", how="inner")
merged_data = merged_data.dropna(subset=["TC_gaze", "Category"])

cm = confusion_matrix(merged_data["TC_gaze"], merged_data["Category"])

if cm.shape == (3, 3):
    TN = cm[1, 1]
    FP = cm[1, 2]
    FN = cm[2, 1]
    TP = cm[2, 2]
else:
    TN, FP, FN, TP = cm.ravel()

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN)

true_label = "TC_gaze"
predicted_label = "Gold_standard"

print(f"Confusion Matrix: {true_label} vs {predicted_label}")
print(f"                      {predicted_label} = 0   {predicted_label} = 1")
print(f" {true_label} = 0    |        {TN}                    {FP}")
print(f" {true_label} = 1    |        {FN}                    {TP}")
print("\nMetrics:")
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"Accuracy: {accuracy:.2f}")

gaze_pred_1 = merged_data[(merged_data["TC_gaze"] == 1)]
gaze_1 = merged_data[(merged_data["Category"] == 1)]

print(f"Total gaze time (Gold Standard) is : {gaze_1.shape[0] * 5} seconds")
print(f"Total gaze time (FLASH) is : {gaze_pred_1.shape[0] * 5} seconds")
