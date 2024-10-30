import re
import os 
import sys
import random
import math
import argparse
import time
import yaml 
import numpy as np 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from pathlib import Path
from glob import iglob
from utils import * 
from gaze_utils import * 
from run_utils import *
from tv_power import *
from main import main

def fix_timestamp_format(timestamp: str) -> str | None:
    if len(timestamp) == 25:  # Check if the timestamp is missing milliseconds based on length
        return (
            timestamp[:-6] + ".000" + timestamp[-6:]
        )  # Add .000 before the timezone info
    elif len(timestamp) < 25:
        raise ValueError
    else:
        return timestamp
    
def get_files_from_folder(folder: str, file_matching_pattern: str, ignore_names: list[str] | None = None):
    if not ignore_names:
        ignore_names = []
    return [
        f
        for f in iglob(f"{folder}/**", recursive=True)
        if os.path.isfile(f)
        and re.search(file_matching_pattern, f)
        and all(ignored not in f for ignored in ignore_names)
    ]


def get_ppt_ids_from_folder(base_path):
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    participant_ids = []
    for folder in folders:
        match = re.match(r'P\d+-(\d+)', folder)
        if match:
            participant_ids.append(match.group(1))
    
    return participant_ids

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default=None, help='path to the main folder')
parser.add_argument('--data_path', type=str, default=None, help='path to the txts folder')
parser.add_argument('--datacsv', type=str, default=None, help='redcap data raw mode file path')

args = parser.parse_args()
ppt_ids = get_ppt_ids_from_folder(args.data_path)

if not ppt_ids:
    raise ValueError("No valid participant IDs found in the data folder.")

print(f"Participant IDs: {ppt_ids}")

for ppt_id in ppt_ids:
    #if ppt_id!='1588':
        #continue

    formatted_ppt_id = f"P1-{ppt_id}"    
    print(f"Processing Participant ID: {formatted_ppt_id}")

    warnings = [
        "Corrupt JPEG data",
        "DeprecationWarning",
        "UserWarning",
        "Deprecated in NumPy 1.20",
        "Failed to load image Python extension",
        "Overload resolution failed:",
        "M is not a numpy array, neither a scalar",
        "Expected Ptr<cv::UMat> for argument",
        "Traceback",
        "warpAffine",
        "nimg = face_align.norm_crop(face_img_bgr, pts5)",
        "facen = model.get_input(face, facelmarks.astype(np.int).reshape(1,5,2), face=True)",
        "face = io.imread(os.path.join(path, fname))",
        "detFacesLog, bboxFaces, idxFaces = pipe_frames_data_to_faces",
        "test_vid_frames_batch_v7_2fps_frminp_newfv_rotate.py",
        "insightface/deploy/face_model.py",
        "insightface/utils/face_align.py",
    ]

    # Add normal strings to this list
    normals = [
        "Loading symbol saved by previous version",
        "Symbol successfully upgraded!",
        "Running performance tests",
        "Resource temporarily unavailable",
        "RTNETLINK answers: File exists",
    ]

    valid_log_paths = get_files_from_folder(args.data_path , fr"{ppt_id}[\s\S]*flash_logstderr")
       

    logs_with_issues = []
    for log_path in valid_log_paths:

        with open(log_path) as log:

            log_lines = log.readlines()

        issues = [
            f"{log_line.strip()}\n"
            for log_line in log_lines
            if log_line
            and all(
                normal_line not in log_line
                for normal_line in normals
            )
        ]

        warnings = [
            issue
            for issue in issues
            if any(
                warning_line in issues
                for warning_line in warnings)
        ]

        errors = [
            issue 
            for issue in issues
            if issue not in warnings
        ]

        #if warnings or errors:
        if errors:
            logs_with_issues.append(log_path)


    study4 = False
    
    if not study4:
        num_days = 10
        redcap_data = pd.read_csv(args.datacsv)
        
        s = redcap_data['ptid']
        if s[s.isin(['P1-' + ppt_id+'-A'])].empty:
            ppt_data = redcap_data[redcap_data['ptid']=='P1-' + ppt_id]    
        else:
            ppt_data = redcap_data[redcap_data['ptid']=='P1-' + ppt_id+'-A']    
        
        start_date = ppt_data.iloc[0]['first']
        tv_count_ls = [0,1,2,3]
        tv_count = ppt_data.iloc[0]['tvcount'] 
        
        tv_count = tv_count_ls[tv_count-1]
       

        with open('%s/input/configs/%s.yaml'%(args.base_path,ppt_id)) as stream:
            tv_config = yaml.safe_load(stream) 
        
    else:
        num_days = 3
        with open('../../tech_data_post_processed/input/configs/%s.yaml'%ppt_id) as stream:
            tv_config = yaml.safe_load(stream)    
        
        start_date = str(tv_config['start_date'])
        tv_count = tv_config['tv_count']

  
    if tv_count > 0:
        if study4:
            tvdev_data, dev_ids, tv_count = get_tv_data_study4(tv_config, tv_count)
        else:
            tvdev_data, dev_ids, tv_count = get_tv_data(ppt_data, tv_config, tv_count)
           
        print('Data path \t: %s'%args.data_path)
        print('Participant-id \t: %s'%ppt_id)    
        print('Start Date \t: %s'%start_date)    
        print('TV count \t: %d'%tv_count)

        dev_ids_ = [str(s) for s in dev_ids]
        print('FLASH dev ids \t: %s'%(','.join(dev_ids_)))

        gaze_df_tvs = []
        summary_dfs = []
        
        for each_tv in range(tv_count):
            tv_data = tvdev_data[each_tv]

            print('######################### Device processing ##############################')
            print('Processing TV-%d, Device ID: %s\n'%(each_tv+1, dev_ids_[each_tv]))
            print(tv_data)
            
            exclude = tv_data['config']['exclude']
            if not exclude:
                if study4:
                    gaze_df_10d, summary_df = main(args.data_path, formatted_ppt_id, start_date, tv_data, num_days, study4=True)
                else:
                    gaze_df_10d, summary_df = main(args.data_path, formatted_ppt_id, start_date, tv_data, num_days, study4=False)

                gaze_df = pd.concat(gaze_df_10d, axis=0)
                gaze_df.apply(fix_timestamp_format)
                
                miss_issue = False
                if not np.isnan(summary_df['miss_tvon'][0]):
                    if any(summary_df['miss_tvon'] >= 20):
                        miss_issue = True
                        miss_dates = summary_df[summary_df['miss_tvon'] >= 20].index
                else:
                    if any(summary_df['miss_time'] >= 20):
                        miss_issue = True
                        miss_dates = summary_df[summary_df['miss_time'] >= 20].index
                
                if miss_issue:
                    print('\n')
                    print('######################### MISSING DATE NOTICED ############################')
                    print('Miss value greater than 20 mins.')
                    print(miss_dates)
                    print('###########################################################################')
                    print('\n')
            else:
                gaze_df = None
                summary_df = None
                print('\n')
                print('######################### EXCLUDING THIS DEVICE ###########################')
                print('The config indicated this device to be excluded. Config notes below.')
                print(tv_data['config']['notes'])
                print('###########################################################################')
                print('\n')

            summary_dfs.append(summary_df)
            gaze_df_tvs.append(gaze_df)
            
        gaze_df_tvs = combine_gaze_tvs(gaze_df_tvs, tv_count, max_num_tvs=3)

        print('\n')
        print('######################### Check the summaries match #######################')
        rand_day_check(start_date, gaze_df_tvs, num_days)
        rand_day_check(start_date, gaze_df_tvs, num_days)
        print('###########################################################################')
        
    else:
        summary_dfs = []
        
        given_date = pd.to_datetime(start_date)
        target_date = given_date + pd.Timedelta(days=num_days)
        
        gz_index = pd.date_range(start=given_date, end=target_date, freq='5S')
        gaze_df_tvs = pd.DataFrame(index=gz_index, columns=['tv1-gaze', 'tv1-exponly', 'tv2-gaze', 'tv2-exponly', 'tv3-gaze', 'tv3-exponly'], data=-1)

    ####################################    
    ########### save logs ##############
    ####################################

    outpath = os.path.join(args.base_path,'output',ppt_id)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    for i, summ in enumerate(summary_dfs):
        if summ is not None:
            fname = 'tv%d_summary.csv'%(i+1)
            path_ = os.path.join(outpath,fname)
            summ = reduce_summary(summ)
            summ.astype(float).to_csv(path_, sep=',',float_format='%.2f')

    fname = '%s_tv_viewing_data.csv'%ppt_id
    path_ = os.path.join(outpath,fname)
    gaze_df_tvs.astype(int).to_csv(path_, sep=',')
        
    tv_info = {}
    tv_info['start_date'] = start_date
    tv_info['family_id'] = int(ppt_id)
    tv_info['tv_count'] = tv_count

    for i in range(tv_count):
        dev_id = int(tvdev_data[i]['id'])
        location = tvdev_data[i]['location']
        notes = tvdev_data[i]['config']['notes']
        excluded = tvdev_data[i]['config']['exclude']
        tv_info['tv%d'%(i+1)] = {'flash_dev_id': dev_id, 'location': location, 'exclude': excluded, 'notes': notes}

    fname = '%s_tv_info.yaml'%ppt_id
    path_ = os.path.join(outpath,fname)
    with open(path_, 'w') as outfile:
        yaml.dump(tv_info, outfile, default_flow_style=False)

    print("\n\n".join(logs_with_issues))
