# flash-post-processing
TV, Mobile, and Physical activity processing and plot codes
> Requirements - pandas, matplotlib, datetime, yaml, numpy

## FLASH-TV post-processing 
**Preparing the data before running the code**
 - Create a folder `post-processed-data` to store results
   - create a folder `input/configs`
   - create a folder `output`
 - Create a folder with just participant ID `1234` at the local path, say `datapath`
   - Download the participant data from the server to this folder
   - Copy each device data zip folder format: `P1-1234XYZ_data.zip` where `XYZ` is the device ID. Extract the zip files
 - In each device data folder:
   - Move all the flash-tv txt logs files to a folder named `txts`
   - Edit the power log `P1-1234XYZ_tv_power_5s.csv`; replace the 2 first text lines with - `power;date;time` as column names
 - Prepare the input config file
   - In the folder `post-processed-data/inputs/configs/`, create a `1234.yaml` file. Take a look at `code/dummy.yaml` to get a sense of the variables
   - This file has a bunch of variables like
     - TV power threshold value
     - setting the power loss timestamps from the notes
     - exclude variable for any device where we decided to exclude the data from a particular TV        

**Setting TV power threshold value for each value**
 - Two steps to set the value of `power_threshold`
    - Read the threshold from the labeled screenshots obtained from visit 1 of the data collection
    - Plot the TV power consumption data using `code/tv_power.py` by setting values of `file_path; start_date`
       - Infer the threshold from these plots.

**Process the logs with `code/run.py`**
 - Download redcap rawdata in csv format from the redcap database
 ```
 python run.py --base_path datapath --ppt_id 1234 --datacsv redcap_rawdata.csv --output path_to post-processed-data
 ```

**Check the outputs and verify the summaries**
 - after running the above code, it should create a folder in `post-processed-data/output/1234/` storing the outputs
    - Inside the participant folder, you will see `1234_tv_info.yaml; 1234_tv_viewing_data.csv; tv1_summary.csv; tv2_summary.csv; tv3_summary.csv`
    - Check `readme_tvdata.txt` to understand the structure of these files
