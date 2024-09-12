import os

import pandas as pd
import numpy as np

from datetime import datetime, timedelta
import datetime as dt

from glob import iglob
import re
import sys
from tkinter import filedialog
import traceback

def filter_unique_samples(df_dts, unique_sampleids):

    filtered_unique_ = [0]
    for i in range(len(unique_sampleids)):
        if i == 0:
            continue

        sample_id = unique_sampleids[i]
        app_use_i = df_dts[df_dts["sample_id"] == sample_id][
            "app_usage_time"
        ].to_numpy()
        # print(i, app_use_i, app_use_i.sum())

        matched = False
        for j in filtered_unique_:
            sample_id = unique_sampleids[j]
            app_use_j = df_dts[df_dts["sample_id"] == sample_id][
                "app_usage_time"
            ].to_numpy()
            # print(j, app_use_j, app_use_j.sum())

            if (
                app_use_i.shape == app_use_j.shape
                and app_use_i.sum() == app_use_j.sum()
            ):
                matched = True
                break

        if matched == False:
            filtered_unique_.append(i)

    filtered_unique_ = [unique_sampleids[i] for i in filtered_unique_]

    return filtered_unique_


def append_data(data, df_dts, sample_id, ppt):

    curr_dts = datetime.strptime(df_dts.iloc[0]["recordeddate"], "%Y-%m-%dT%H:%M:%S%z")
    start_ts = None
    prev_delta = 0
    idx = 0

    for i, row in df_dts.iterrows():
        if row["sample_id"] == sample_id:
            # delta = dt.timedelta(seconds=int(row['app_usage_time']))
            delta = int(row["app_usage_time"])

            start_ts = curr_dts + dt.timedelta(seconds=prev_delta)  # prev_delta
            stop_ts = start_ts + dt.timedelta(seconds=delta)  # delta

            # print(idx, row['recordeddate'], str(start_ts)[:10], str(start_ts)[11:-6], str(stop_ts)[11:-6], row['app_category'], row['bundle_identifier'])
            idx = idx + 1

            data["sample_id"].append(row["sample_id"])
            data["recordeddate"].append(row["recordeddate"])
            data["date"].append(str(start_ts)[:10])
            data["start_time"].append(str(start_ts)[11:-6])
            data["stop_time"].append(str(stop_ts)[11:-6])
            data["duration"].append(row["app_usage_time"])
            data["participant_id"].append(str(ppt))
            data["user"].append("Target Child")
            data["app_category"].append(row["app_category"])
            data["app_namedetail"].append(row["bundle_identifier"])
            prev_delta = prev_delta + delta

    return data


def process_raw_Chronicle_iPhone_sensor_data_files(
    *args,
    Chronicle_iPhone_sensor_data_folder: str,
    ignore_list: list,
    specific_date: str = None,
    output_folder: str,
    **kwargs,
):

    Chronicle_iPhone_sensor_data_files = [
        f
        for f in iglob(f"{Chronicle_iPhone_sensor_data_folder}/**", recursive=True)
        if os.path.isfile(f)
        and re.search(r"([\s\S]*P[1|3]-[1|3]\d{3}[\s\S]*).csv", f)
        and all(ignored not in f for ignored in ignore_list)
    ]

    Chronicle_iPhone_sensor_data_files.sort()

    for file in Chronicle_iPhone_sensor_data_files:
       
        try:
            df = pd.read_csv(file)
            df = df[df["sensor_type"] == "deviceUsage"]
            df = df[~pd.isna(df["app_usage_time"])]
            unique_datetimestamp = df["recordeddate"].unique()
            participant_id = str(df.iloc[1]["participant_id"])

            #print("Participant ID: ", participant_id)

            # user, appshortname, start, stop, appdetailedname
            data = {
                "sample_id": [],
                "recordeddate": [],
                "date": [],
                "start_time": [],
                "stop_time": [],
                "duration": [],
                "participant_id": [],
                "user": [],
                "app_category": [],
                "app_namedetail": [],
            }

            for u_dts in unique_datetimestamp:
                # get the usage in the segment

                df_dts = df[df["recordeddate"] == u_dts]
                total_unlock_duration = df_dts.iloc[0]["total_unlock_duration"]
                sample_duration = df_dts.iloc[0]["sample_duration"]

                if any(df_dts["sensor_type"] == "messagesUsage"):
                    raise "messageUsage detected!!"

                if total_unlock_duration > sample_duration:
                    raise "sample duration exceeds total_unlock_duration!!"

                unique_sampleids = df_dts["sample_id"].unique()
                filtered_unique_ = filter_unique_samples(
                    df_dts, unique_sampleids
                )  # removes duplicates across sample_ids
                # unique_df = filter_unique_appuses(df_dts, filtered_unique_) removes duplicates within the sample_ids

                # removes duplicates within the sample_ids
                df_dts = df_dts.drop_duplicates(
                    subset=["sample_id", "app_usage_time", "bundle_identifier"],
                    keep="first",
                ).reset_index(drop=True)

                # print(df_dts['app_usage_time'])

                total_ = 0
                for num_, sample_id in enumerate(filtered_unique_):

                    sum_duration = df_dts[df_dts["sample_id"] == sample_id]["app_usage_time"].sum()

                    text_duration = df_dts[df_dts["sample_id"] == sample_id]["text_input_duration"]

                    #text_duration[np.isnan(text_duration)] = 0

                    sum_text_duration = 0  # text_duration.sum()

                    if total_ + sum_duration <= (total_unlock_duration * 1.1) or num_ == 0:
                        total_ = total_ + sum_duration
                        data = append_data(data, df_dts, sample_id, participant_id)
                    else:
                        break

                # if abs(total_unlock_duration-total_)>10:
                # print(total_unlock_duration, total_,)# filtered_unique_,) #unique_sampleids)

            new_df = pd.DataFrame(data)
            new_df["participant_id"] = participant_id

            cols = [
                "sample_id",
                "participant_id",
                "date",
                "recordeddate",
                "start_time",
                "stop_time",
                "duration",
                "user",
                "app_category",
                "app_namedetail",
            ]

            new_df = new_df[cols]
            new_df["date"] = pd.to_datetime(new_df["date"]).dt.date
            new_df["recordeddate"] = pd.to_datetime(
                new_df["recordeddate"], utc=True
            ).dt.tz_convert(
                "America/Chicago"
            )
            new_df = new_df.sort_values(by="recordeddate")

            # print(new_df.iloc[350:375])

            # fix the date mismatch
            idx_match = new_df[new_df["recordeddate"].dt.date != new_df["date"]].index
            for idx in idx_match:
                # print('check')
                # print(new_df.loc[idx])
                # new_df.loc[idx,'date'] =  new_df.loc[idx,'recordeddate'].date()
                new_df.loc[idx, "recordeddate"] = pd.to_datetime(new_df.loc[idx, "date"])
                # new_df.loc[idx,'start_time'] = '00:00:00'
                # print(new_df.loc[idx])

            new_df = new_df.sort_values(by="recordeddate")
            # print(new_df.iloc[350:375])
            new_df["recordeddate"] = new_df["recordeddate"].dt.time

            new_df.rename(
                columns={
                    "user": "username",
                    "start_time": "start_timestamp",
                    "stop_time": "stop_timestamp",
                    "recordeddate": "epoch_window",
                },
                inplace=True,
            )

            os.makedirs(output_folder, exist_ok=True)

            if specific_date is not None:
                cat_usels = []
                new_df = new_df[new_df["date"] == specific_date]
                app_categories = new_df["app_category"].unique()
                for each_cat in app_categories:
                    cat_df = new_df[new_df["app_category"] == each_cat]
                    cat_use = cat_df["duration"].sum() / 60.0
                    print("%s Use: \t\t %.1f mins" % (each_cat, cat_use))
                    cat_usels.append(cat_use)

                cat_usels = np.array(cat_usels)
                # print("Total Use: %.1f mins" % (new_df["duration"].sum() / 60.0))
                # print(cat_usels.sum())
                file_name = f"{output_folder}/{participant_id}-A-T1 Chronicle iOS Processed Data for {specific_date} Only.csv"
                new_df.to_csv(file_name, index=False)
                # print("results saved at: ", file_name)
            else:
                # if "P1-" in participant_id and "-A" not in participant_id:
                #     save_name = (
                #         f"{output_folder}/{participant_id}-A-T1 Chronicle iOS Processed Data.csv"
                #     )
                # else:
                #     save_name = (
                #             f"{output_folder}/{participant_id} Chronicle iOS Processed Data.csv"
                #     )
                save_name = (
                    f"{output_folder}/{participant_id} Chronicle iOS Processed Data.csv"
                )
                new_df.to_csv(save_name, index=False)
                # print("results saved at: ", save_name)
            print(f"Finished processing Chronicle iPhone sensor data for participant {participant_id}")
        except:
            print(f"An error occurred while processing Chronicle iPhone sensor data for participant {participant_id}: {traceback.format_exc()}")
            continue


if __name__ == "__main__":
    
    IGNORE_LIST = ["Do Not Use", "Archive"]
    SCRIPT_FOLDER = os.path.dirname(sys.argv[0])
    OUTPUT_FOLDER = f"{SCRIPT_FOLDER}/Processed Chronicle iPhone Sensor Data"
    CHRONICLE_IPHONE_SENSOR_DATA_FOLDER = filedialog.askdirectory() # Or specify path manually

    try:
        assert (CHRONICLE_IPHONE_SENSOR_DATA_FOLDER != None) & (CHRONICLE_IPHONE_SENSOR_DATA_FOLDER != "")
    except AssertionError:
        print("Please select the Chronicle iPhone sensor data folder using the folder dialog, or specify the path to the folder manually.")
        exit()

    process_raw_Chronicle_iPhone_sensor_data_files(
        Chronicle_iPhone_sensor_data_folder = CHRONICLE_IPHONE_SENSOR_DATA_FOLDER,
        ignore_list = IGNORE_LIST,
        specific_date = None,
        output_folder = OUTPUT_FOLDER,
    )