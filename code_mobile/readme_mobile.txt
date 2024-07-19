###########################################
####### File Names                  #######
###########################################

----> FileName: 568_android_final.csv, 579_ipad_final.csv, 581_ios_final.csv
    
    -> 568 is the family/participant ID
    -> android indicates it is an Android device and Chronicle android was used to measure screentime
    -> ios indicates it is an iPhone device and Chronicle iOS was used to measure screentime
    -> ipad indicates it is an iPad device and screentime screenshots are processed to measure screentime

###########################################
####### Android file structure       ######
###########################################

----> FileName: 568_android_final.csv

    -> Contains Mobile screen use, with apps start and stop time of the usage

        COLUMN Names: event_timestamp, study_id, participant_id, date, start_timestamp, stop_timestamp, username, application_label, app_package_name, timezone
        
        Explanation for the individual columns below
        
        1) event_timestamp      : indicates the exact time stamp of when the app usage started
        2) study_id             : indicates the study id from chronicle
        3) participant_id       : indicates the participant ID
        4) date                 : indicates date of the app usage data
        5) start_timestamp      : indicates exact start time of the app usage
        6) stop_timestamp       : indicates exact stop time of the app usage
        7) username             : indicates who the user is - "Target Child", "Other", "None"
                                  None - Unknown user resulting from ignoring the user identification notifications
        8) application_label    : indicates short name of the app
        9) app_package_name     : indicates detailed name of the app
        10) timezone            : indicates local timezone
    
###########################################
#######  iOS (iPhone) file structure ######
###########################################

----> FileName: 581_iphone_final.csv

    -> Contains Mobile screen use, with apps start and stop time of the usage
    -> NOTE: For iPhones, the exact start and stop times of the app use are not available, only the durations are reported within a 15 minute epoch

        COLUMN Names: event_timestamp, sample_id, participant_id, date, epoch_window, start_timestamp, stop_timestamp, duration, username, app_category, app_namedetail
        
        Explanation for the individual columns below
        
        1) event_timestamp      : indicates the simulated time stamp of when the app usage started
        2) sample_id            : indicates the sample id from iOs might be useful in the future
        3) participant_id       : indicates the participant ID
        4) date                 : indicates date of the app usage data
        5) epoch_window         : indicates time stamp of the beginging of the 15 mins window 
        6) start_timestamp      : indicates simulated start time of the app usage
        7) stop_timestamp       : indicates simulated stop time of the app usage
        8) duration             : indicates duration of the app use in seconds with in the 15 mins window
        9) username             : indicates who the user is - "Target Child"
                                  NOTE: there is no shared use so all use is target child's
        10) app_category        : indicates short name of the app
        11) app_namedetail      : indicates app name details in code which might be useful in the future
    
        
###########################################
#######    iPad file structure       ######
###########################################
        
----> FileName: 579_ipad_final.csv

    -> Contains Mobile screen use, with apps start and stop time of the usage
    -> NOTE: For iPads, the exact start and stop times of the app use are not available, only the durations are reported within a 1 hour epoch

        COLUMN Names: event_timestamp, participant_id, date, epoch_window, start_timestamp, stop_timestamp, username, application_label
        
        Explanation for the individual columns below
        
        1) event_timestamp      : indicates the simulated time stamp of when the app usage started
        2) participant_id       : indicates the participant ID
        3) date                 : indicates date of the app usage data
        4) epoch_window         : indicates time stamp of the beginging of the 1 hour window 
        5) start_timestamp      : indicates simulated start time of the app usage
        6) stop_timestamp       : indicates simulated stop time of the app usage
        7) username             : indicates who the user is - "Target Child"
                                  NOTE: there is no shared use so all use is target child's
        8) application_label   : indicates short name of the app
        
