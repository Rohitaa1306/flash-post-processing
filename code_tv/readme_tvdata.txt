###########################################
####### TVs INFORMATION             #######
###########################################

----> FileName: 1147_tv_info.yaml
    
    -> 1147 is the family/participant ID
    -> Text file containing TV details
        
        family_id   : participant ID
        start_date  : start_date of the data collection
        tv_count    : indicates number of TVs in the home
        flash ver.  : indicates FLASH version used 
        
        # for each tv we indicate the FLASH device id, and location
        tv1: 1st TV details
            exclude         : [True/False] # If true, this TV data is excluded due to reasons like FLASH-TV malfunction or other reasons. More details are in the associated notes below.     
            flash_dev_id    : [FLASH device ID]
            location        : [1st TV location]
            notes           : Includes notes indicating anything abnormal like TV power plug not working or reasons for excluding this TV.
        tv2: 2nd TV details
            exclude         : [True/False] # If true, this TV data is excluded due to reasons like FLASH-TV malfunction or other reasons. More details are in the associated notes below.     
            flash_dev_id    : [FLASH device ID]
            location        : [1st TV location]
            notes           : Includes notes indicating anything abnormal like TV power plug not working or reasons for excluding this TV.
            
###########################################
####### Time Stamped TV viewing data ######
###########################################

----> FileName: 1147_tv_viewing_data.csv
    
    -> 1147 is the family/participant ID
    -> Contains tv viewing information for 3 TVs with the below indicated columns
        
        COLUMN Names: dateTimeStamp, tv1-gaze, tv1-exponly, tv2-gaze, tv2-exponly, tv3-gaze, tv3-exponly
        Explanation for the individual columns below
        
        1) dateTimeStamp: indicates the specific datetime epoch, sample indicated below
            sample:
            2023-11-10 00:00:00
            2023-11-10 00:00:05
            2023-11-10 00:00:10
            
        2) tv1-gaze: contains 1st TV viewing behaviors, with labels indicated below
             "Gaze"                 : 1
             "No-Gaze"              : 0
             "Missing"              : 2     (This can happen due to flash device restarting / rebooting / malfunctioning)
             "Flash-powered off"    : 3
             "Not-present"          : -1    (TV is not there)
        3) tv1-exponly: contains 1st TV exposure only behaviors, with labels indicated below 
             "Exposure"             : 1
             "No-Exposure"          : 0
             "Missing"              : 2     (This can happen due to flash device restarting / rebooting / malfunctioning)
             "Flash-powered off"    : 3
             "Not-present"          : -1    (TV is not there)
                         
        4) tv2-gaze: data for the 2nd TV 
            if there is no 2nd TV, it will labeled with "Not-present" (=-1)
        5) tv2-exponly: data for the 2nd TV
            if there is no 2nd TV, it will labeled with "Not-present" (=-1)
        
        6) tv3-gaze: data for the 3rd TV
            if there is no 3rd TV, it will labeled with "Not-present" (=-1)
        7) tv3-exponly: data for the 3rd TV
            if there is no 3rd TV, it will labeled with "Not-present" (=-1)    
        
###########################################
####### TV Summary data              ######
###########################################
For each TV, up to the number of TVs we have a summary csv file. The structure of the file is indicated below. Use the summary to cross check / verify the summed up gaze data from the above 1147_tv_viewing_data.csv file. 
    -> tv1_summary.csv
    -> tv2_summary.csv
    
----> FileName: tv1_summary.csv
    
    -> Contains 1st TV's gaze and exposure time in minutes day wise for the 10 days of the data collection
        
        COLUMN Names: Date,miss_tvon,gaze_tvon,exp_tvon
        Explanation for the individual columns below
        
        1) Date: indicates the date of the data collection
            sample:
            2023-11-10
            2023-11-11
            .
            .
            2023-11-19
        
        2) miss_tvon: indicates the amount of missing data in minutes due to FLASH device restarting / rebooting / malfunctioning
            This value is generally in the range of 10-15 mins per day
            
        3) gaze_tvon: indicates the amount of TV watching time (minutes) when the TV is ON
        
        4) exp_tvon: indicates the amount of TV exposure only time (minutes) when the TV is ON
            exposure is counted as when the child is detected in the room with the TV ON
        
                
----> FileName: tv2_summary.csv
    
    -> Contains 2nd TV's gaze and exposure time in minutes day wise for the 10 days of the data collection
        
        COLUMN Names: Date,miss_tvon,gaze_tvon,exp_tvon
        Explanation for the individual columns are same as above

----> FileName: tv3_summary.csv
    
    -> Contains 3rd TV's gaze and exposure time in minutes day wise for the 10 days of the data collection
        
        COLUMN Names: Date,miss_tvon,gaze_tvon,exp_tvon
        Explanation for the individual columns are same as above
        
