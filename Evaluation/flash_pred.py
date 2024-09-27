import pandas as pd

file_path = 'C:\\Users\\u255769\\Downloads\\Evaluation\\606_flash_tvdata.txt'
df = pd.read_csv(file_path)
gz_df = df[df['flash_prediction'] == 1]
print(len(gz_df))

file_path_1 = 'C:\\Users\\u255769\\Downloads\\Evaluation\\output.csv'
df_1 = pd.read_csv(file_path_1)
gz_df_1 = df_1[df_1['TC_gaze'] == 1]
filtered_gz_df_1 = gz_df_1[gz_df_1['dateTimeStamp'] <= '2023-09-25 14:01:52']
print(len(filtered_gz_df_1))


