import os 
import sys

import numpy as np 
import pandas as pd
import yaml


#path = '/home/akv/FLASH_PO1/flash-tv-post-processing/data/output'
path = '/home/akv/FLASH_PO1/tech_data_post_processed/study4_tv_output'
num_days=3

ppts = os.listdir(path)
ppts.sort()

for ppt in ppts:
    
    # read yaml
    with open('%s/%s/%s_tv_info.yaml'%(path,ppt,ppt)) as stream:
        tv_info = yaml.safe_load(stream)    
    
    tv_count = tv_info['tv_count']
    
    #print(tv_info)
    p = '%d,%s,%d'%(tv_info['family_id'], tv_info['start_date'], tv_info['tv_count'])
    p2 = '%s,%s,%s'%(tv_info['tv1']['location'], tv_info['tv1']['exclude'], tv_info['tv1']['notes'])
    if tv_count==2:
        p3 = '%s,%s,%s'%(tv_info['tv2']['location'], tv_info['tv2']['exclude'], tv_info['tv2']['notes'])
    else:
        p3 = '%s,%s,%s'%(None,None,None)
        
    print(p+','+p2+','+p3)
