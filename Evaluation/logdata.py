import os
import pandas as pd
import numpy as np
from datetime import datetime
from gaze_estimate import correct_rotation, convert_to_gaze, convert_lims

def main(base_path, ppt_id):
    
    reg_file = os.path.join(base_path, f"{ppt_id}_reg.txt")   
    reg_df = pd.read_csv(reg_file, sep=r'\s+')
    reg_df['dateTimeStamp'] = reg_df['date'] + ' ' + reg_df['TimeStamp']
    reg_df = reg_df.drop(columns=['date', 'TimeStamp'])    
    reg_df['dateTimeStamp'] = pd.to_datetime(reg_df['dateTimeStamp'])
    reg_df.set_index('dateTimeStamp', inplace=True)
    reg_df = reg_df.sort_index()
    reg_df.index = reg_df.index.floor('S')

    rot_file = os.path.join(base_path, f"{ppt_id}_rot.txt")
    rot_df = pd.read_csv(rot_file, sep=r'\s+')
    rot_df['dateTimeStamp'] = rot_df['date'] + ' ' + rot_df['TimeStamp']
    rot_df['dateTimeStamp'] = pd.to_datetime(rot_df['dateTimeStamp'])
    rot_df = rot_df.drop(columns=['date', 'TimeStamp'])    
    rot_df.set_index('dateTimeStamp', inplace=True)
    rot_df = rot_df.sort_index()
    rot_df.index = rot_df.index.floor('S')
    
    index_match = rot_df.index.equals(reg_df.index)
    assert index_match

    start_date_str = str(reg_df.index[0])
    end_date_str = str(reg_df.index[-1])
    gz_index = pd.date_range(start=start_date_str, end=end_date_str, freq='s')
    gz_df = pd.DataFrame(index=gz_index)
    gz_df['TC_gaze'] = 5
    gz_df['TC_exposure_only'] = 5

    df_ = reg_df[reg_df['rot.'].abs()>=30] 
    if df_.shape[0]>0:
        phi_ = df_[['phi','theta','rot.']].values 
        phi_corrected = correct_rotation(phi_)
        reg_df.loc[df_.index,['phi','theta']] = phi_corrected[:,:2]
    
    df_ = rot_df[rot_df['tag']=='Gaze-no-det']
    assert df_['tcPresent'].sum()==0
    
    gz_df.loc[df_.index,'TC_gaze'] = 0
    inc_index = df_.index + pd.Timedelta(seconds=1)
    gz_df.loc[inc_index,'TC_gaze'] = 0 # cause FLASHtv takes frames 2-sec and gives one prediction
    
    gz_df.loc[df_.index,'TC_exposure_only'] = 0
    gz_df.loc[inc_index,'TC_exposure_only'] = 0 # cause FLASHtv takes frames 2-sec and gives one prediction
    
    del df_

    #1 = Gz (flash pred) Gaze-det, TC present
    #0 = No-Gz (flash pred) Gaze-det, TC present (TC exposure only)
    df_1 = rot_df[rot_df['tag']=='Gaze-det']
    df_2 = reg_df[rot_df['tag']=='Gaze-det']
    assert len(df_1)==len(df_2)
    
    gz_data = df_1[['phi','theta','top','left','bottom','right']].values
    gz_reg = df_1[['phi','theta']].values

    mixmodel = True
    if mixmodel:
        gz_data[:,:2] = 0.6*gz_data[:,:2] + 0.4*gz_reg

    lims = convert_lims({'size': 32.0, 'cam_height': 43.0, 'tv_height': 50.0, 'view_dist': 64.0})
    pred_gz = convert_to_gaze(gz_data, lims).astype(np.int32)
    tc_exp_only = 1 - pred_gz
    
    assert pred_gz.shape[0] == gz_data.shape[0]
    assert (pred_gz==2).sum() == 0
    assert (pred_gz<0).sum() == 0
    assert (pred_gz>1).sum() == 0
    
    gz_df.loc[df_1.index,'TC_gaze'] = pred_gz
    inc_index = df_1.index + pd.Timedelta(seconds=1)
    
    try:
        gz_df.loc[inc_index,'TC_gaze'] = pred_gz[:len(inc_index)]
    except KeyError:
        pass  

    gz_df.loc[df_1.index,'TC_exposure_only'] = tc_exp_only

    try:
        gz_df.loc[inc_index,'TC_exposure_only'] = tc_exp_only[:len(inc_index)]
    except KeyError:
        pass 
    gz_df[gz_df==5] = 0
  
    gz_df = gz_df.loc[reg_df.index]

    gz_df_1 = gz_df[gz_df['TC_gaze'] == 1]
    print(len(gz_df_1))
    
    return gz_df

if __name__ == "__main__":
    base_path = 'C:\\Users\\u255769\\Downloads\\Evaluation\\txts'
    ppt_id = 606
    ppt_id = str(ppt_id)
    gaze_dfs = main(base_path, ppt_id)
    gaze_dfs.to_csv('C:\\Users\\u255769\\Downloads\\Evaluation\\output.csv')
