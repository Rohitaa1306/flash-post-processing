import os
import pandas as pd
import numpy as np
from datetime import datetime
from gaze_estimate import correct_rotation, convert_to_gaze, convert_lims

def main(base_path, ppt_id):
    
    reg_file = os.path.join(base_path, f"{ppt_id}_reg.txt")   
    reg_df = pd.read_csv(reg_file, sep=r'\s+')
    reg_df['dateTimeStamp'] = reg_df['date'] + ' ' + reg_df['TimeStamp']
    reg_df['dateTimeStamp'] = pd.to_datetime(reg_df['dateTimeStamp'], format="%Y-%m-%d %H:%M:%S.%f")
    reg_df = reg_df.drop(columns=['date', 'TimeStamp'])    
    reg_df.set_index('dateTimeStamp', inplace=True)
    reg_df = reg_df.sort_index()

    rot_file = os.path.join(base_path, f"{ppt_id}_rot.txt")
    rot_df = pd.read_csv(rot_file, sep=r'\s+')
    rot_df['dateTimeStamp'] = rot_df['date'] + ' ' + rot_df['TimeStamp']
    rot_df['dateTimeStamp'] = pd.to_datetime(rot_df['dateTimeStamp'], format="%Y-%m-%d %H:%M:%S.%f")
    rot_df = rot_df.drop(columns=['date', 'TimeStamp'])    
    rot_df.set_index('dateTimeStamp', inplace=True)
    rot_df = rot_df.sort_index()
                                                                                                                                                                                                                                                                    
    df_ = reg_df[reg_df['rot.'].abs() >= 30]
    if df_.shape[0] > 0:
        phi_ = df_[['phi', 'theta', 'rot.']].values
        phi_corrected = correct_rotation(phi_)
        reg_df.loc[df_.index, ['phi', 'theta']] = phi_corrected[:, :2]

    start_date_str = str(reg_df.index[0])
    end_date_str = str(reg_df.index[-1])
    gz_index = pd.date_range(start=start_date_str, end=end_date_str, freq='s')
    gz_df = pd.DataFrame(index=gz_index)
    gz_df['TC_gaze'] = 5
    gz_df['TC_exposure_only'] = 5

    rot_df.index = pd.to_datetime(rot_df.index).floor('s')
    reg_df.index = pd.to_datetime(reg_df.index).floor('s')
    reg_df = reg_df.reindex(rot_df.index, fill_value=np.nan)
    
    df_no_det = rot_df[rot_df['tag'] == 'Gaze-no-det']
    common_no_det_index = gz_df.index.intersection(df_no_det.index)
    gz_df.loc[common_no_det_index, 'TC_gaze'] = 0
    gz_df.loc[common_no_det_index, 'TC_exposure_only'] = 0
     
    del df_no_det

    df_det = rot_df[rot_df['tag'] == 'Gaze-det']
    reg_det = reg_df[rot_df['tag'] == 'Gaze-det']
    
    if len(df_det) == len(reg_det):
        df_det = df_det[:-1]
        gz_data = df_det[['phi', 'theta', 'top', 'left', 'bottom', 'right']].values
        gz_reg = df_det[['phi', 'theta']].values
       
        mixmodel = True
        if mixmodel:
            gz_data[:, :2] = 0.6 * gz_data[:, :2] + 0.4 * gz_reg
        
        df_det.index = df_det.index.floor('s')
        gz_df.index = gz_df.index.floor('s')
      
        lims = convert_lims({'size': 32.0, 'cam_height': 43.0, 'tv_height': 50.0, 'view_dist': 64.0})
        pred_gz = convert_to_gaze(gz_data, lims).astype(np.int32)
        
        tc_exp_only = 1 - pred_gz

        assert pred_gz.shape[0] == gz_data.shape[0]
        assert (pred_gz==2).sum() == 0
        assert (pred_gz<0).sum() == 0
        assert (pred_gz>1).sum() == 0

        gz_df.loc[df_det.index,'TC_gaze'] = pred_gz
        gz_df.loc[df_det.index,'TC_exposure_only'] = tc_exp_only
        
    gz_df[gz_df == 5] = 0

    tv_data = gz_df[['TC_gaze', 'TC_exposure_only']].values
    tv_time_sec = (tv_data[:, 0] == 1).sum() 
    tv_exp_only_sec = (tv_data[:, 1] == 1).sum() 

    summary_df = pd.DataFrame({
        'gaze_time': [tv_time_sec],
        'exposure_only_time': [tv_exp_only_sec]
    })

    return gz_df, summary_df

if __name__ == "__main__":
    base_path = 'C:\\Users\\u255769\\Downloads\\Evaluation\\txts'
    ppt_id = 606
    ppt_id = str(ppt_id)
    gaze_dfs, summary_df = main(base_path, ppt_id)
    print(summary_df)
    gaze_dfs.to_csv('C:\\Users\\u255769\\Downloads\\Evaluation\\output.csv')
