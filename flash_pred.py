import pandas as pd

file_path = 'C:\\Users\\u255769\\Downloads\\Evaluation\\606_flash_tvdata.txt'
df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
flash_1_df = df[df['flash_prediction'] == 1]

if not flash_1_df.empty:
    total_time = (flash_1_df['timestamp'].iloc[-1] - flash_1_df['timestamp'].iloc[0]).total_seconds()
else:
    total_time = 0  

print(f"Total time where flash_prediction was 1: {total_time} seconds")
