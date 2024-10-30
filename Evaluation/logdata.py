from imports import *

def epoch_vote(arr):
    votes = [(arr == 0).sum(), (arr == 1).sum(), (arr == 2).sum(), (arr == 3).sum()]
    idx = np.argmax(votes)
    val = votes[idx]

    if val >= 3:
        return idx
    else:
        return 0

def condense_epc(tv_time, tv_exp_only):
    assert tv_time.shape == tv_exp_only.shape
    # assert tv_time.size == 86400

    trim_size = tv_time.size % 5
    if trim_size != 0:
        tv_time = tv_time[:-trim_size]
        tv_exp_only = tv_exp_only[:-trim_size]

    tv_time = tv_time.reshape(-1, 5)
    tv_exp_only = tv_exp_only.reshape(-1, 5)

    tv_time_epc = np.apply_along_axis(epoch_vote, axis=1, arr=tv_time)
    tv_exp_only_epc = np.apply_along_axis(epoch_vote, axis=1, arr=tv_exp_only)
    return tv_time_epc, tv_exp_only_epc

def main(base_path, ppt_id):
    reg_file = os.path.join(base_path, f"{ppt_id}_reg.txt")
    reg_df = pd.read_csv(reg_file, sep=r"\s+")
    reg_df["dateTimeStamp"] = reg_df["date"] + " " + reg_df["TimeStamp"]
    reg_df = reg_df.drop(columns=["date", "TimeStamp"])
    reg_df["dateTimeStamp"] = pd.to_datetime(reg_df["dateTimeStamp"])
    reg_df.set_index("dateTimeStamp", inplace=True)
    reg_df = reg_df.sort_index()
    reg_df.index = reg_df.index.floor("S")

    rot_file = os.path.join(base_path, f"{ppt_id}_rot.txt")
    rot_df = pd.read_csv(rot_file, sep=r"\s+")
    rot_df["dateTimeStamp"] = rot_df["date"] + " " + rot_df["TimeStamp"]
    rot_df["dateTimeStamp"] = pd.to_datetime(rot_df["dateTimeStamp"])
    rot_df = rot_df.drop(columns=["date", "TimeStamp"])
    rot_df.set_index("dateTimeStamp", inplace=True)
    rot_df = rot_df.sort_index()
    rot_df.index = rot_df.index.floor("S")

    index_match = rot_df.index.equals(reg_df.index)
    assert index_match

    start_date_str = str(reg_df.index[0])
    end_date_str = str(reg_df.index[-1])
    gz_index = pd.date_range(start=start_date_str, end=end_date_str, freq="s")
    gz_df = pd.DataFrame(index=gz_index)
    gz_df["TC_gaze"] = 5
    gz_df["TC_exposure_only"] = 5

    df_ = reg_df[reg_df["rot."].abs() >= 30]
    if df_.shape[0] > 0:
        phi_ = df_[["phi", "theta", "rot."]].values
        phi_corrected = correct_rotation(phi_)
        reg_df.loc[df_.index, ["phi", "theta"]] = phi_corrected[:, :2]

    df_ = rot_df[rot_df["tag"] == "Gaze-no-det"]
    assert df_["tcPresent"].sum() == 0

    gz_df.loc[df_.index, "TC_gaze"] = 0
    inc_index = df_.index + pd.Timedelta(seconds=1)
    gz_df.loc[inc_index, "TC_gaze"] = 0  # cause FLASHtv takes frames 2-sec and gives one prediction

    gz_df.loc[df_.index, "TC_exposure_only"] = 0
    gz_df.loc[inc_index, "TC_exposure_only"] = 0  # cause FLASHtv takes frames 2-sec and gives one prediction

    del df_

    # 1 = Gz (flash pred) Gaze-det, TC present
    # 0 = No-Gz (flash pred) Gaze-det, TC present (TC exposure only)
    df_1 = rot_df[rot_df["tag"] == "Gaze-det"]
    df_2 = reg_df[rot_df["tag"] == "Gaze-det"]
    assert len(df_1) == len(df_2)

    gz_data = df_1[["phi", "theta", "top", "left", "bottom", "right"]].values
    gz_reg = df_1[["phi", "theta"]].values

    mixmodel = True
    if mixmodel:
        gz_data[:, :2] = 0.6 * gz_data[:, :2] + 0.4 * gz_reg

    lims = convert_lims({"size": 32.0, "cam_height": 43.0, "tv_height": 50.0, "view_dist": 64.0})
    pred_gz = convert_to_gaze(gz_data, lims).astype(np.int32)
    tc_exp_only = 1 - pred_gz

    assert pred_gz.shape[0] == gz_data.shape[0]
    assert (pred_gz == 2).sum() == 0
    assert (pred_gz < 0).sum() == 0
    assert (pred_gz > 1).sum() == 0

    gz_df.loc[df_1.index, "TC_gaze"] = pred_gz
    inc_index = df_1.index + pd.Timedelta(seconds=1)
    #print(inc_index)
    #print(gz_df)
    #print((gz_df["TC_gaze"] == 1).sum())
    
    inc_index = inc_index[:-1]
    gz_df.loc[inc_index, "TC_gaze"] = pred_gz[: len(inc_index)]
    
    #print((gz_df["TC_gaze"] == 1).sum())

    gz_df.loc[df_1.index, "TC_exposure_only"] = tc_exp_only
    gz_df.loc[inc_index, "TC_exposure_only"] = tc_exp_only[: len(inc_index)]
    
    gz_df[gz_df == 5] = 0
    # print(gz_df)

    tv_data = gz_df[["TC_gaze", "TC_exposure_only"]].values
    # print(tv_data)

    tv_time_sec = (tv_data[:, 0] == 1).sum() / 60.0
    tv_exp_only_sec = (tv_data[:, 1] == 1).sum() / 60.0
    # print(tv_time_sec)
    # print(tv_exp_only_sec)

    tv_time = tv_data[:, 0]
    tv_exp_only = tv_data[:, 1]
    # print(tv_time.shape)
    # print(tv_exp_only.shape)

    tv_time_epc, tv_exp_only_epc = condense_epc(tv_time, tv_exp_only)
    # print(tv_time_epc.shape)
    # print(tv_exp_only_epc.shape)

    # flash per epoch tv data
    gz_epoch_index = pd.date_range(start=start_date_str, end=end_date_str, freq="5S")
    gz_epc_df = pd.DataFrame(index=gz_epoch_index)
    #print(gz_epc_df)

    if len(gz_epoch_index) != len(tv_time_epc):
        min_len = min(len(gz_epoch_index), len(tv_time_epc))
        gz_epoch_index = gz_epoch_index[:min_len]
        tv_time_epc = tv_time_epc[:min_len]
        tv_exp_only_epc = tv_exp_only_epc[:min_len]

    gz_epc_df.loc[gz_epoch_index, "TC_gaze"] = tv_time_epc
    gz_epc_df.loc[gz_epoch_index, "TC_exposure_only"] = tv_exp_only_epc


    # print(gz_epc_df.shape)

    tt = (tv_time_epc == 1).sum() * 5 / 60.0
    eo = (tv_exp_only_epc == 1).sum() * 5 / 60.0

    # print('TV time: \t%.2f'%tt)
    # print('TV exponly: \t%.2f'%eo)

    return gz_df, gz_epc_df

if __name__ == "__main__":
    base_path = "C:\\Users\\u255769\\flash-post-processing\\Evaluation\\txts"
    ppt_id = 606
    ppt_id = str(ppt_id)
    gaze_dfs_1, gaze_dfs_5 = main(base_path, ppt_id)
    gaze_dfs_1.index.name = 'timestamp'
    gaze_dfs_5.index.name = 'timestamp'
    gaze_dfs_1.to_csv("C:\\Users\\u255769\\flash-post-processing\\Evaluation\\output_1.csv")
    gaze_dfs_5.to_csv("C:\\Users\\u255769\\flash-post-processing\\Evaluation\\output_5.csv")
