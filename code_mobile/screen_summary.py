import numpy as np
import matplotlib.pyplot as plt
import sys

from datetime import datetime
import datetime as dt

def convert(time):
    h,m,s = time.split(':')
    h = int(h)
    m = int(m)
    s = int(s[:2])
    
    
    n = h + m/60.0 + s/3600.0
    #print(time, h,m,s, n)
    return n


def convert_secs(time):
    h,m,s = time.split(':')
    h = int(h)
    m = int(m)
    s = int(s[:2])
    
    
    n = h*3600 + m*60 + s
    #print(time, h,m,s, n)
    return n

def convert_fromsecs(numsecs):
    h = numsecs // 3600
    ms = numsecs % 3600
    m = ms // 60
    s = ms % 60
    
    n = h + m/60.0 + s/3600.0
    #print(time, h,m,s, n)
    return n

def process(lines):
    data = [l.split(',') for l in lines]
    data = [[convert(d[0].split(' ')[1]), int(d[-2]), int(d[-1])] for d in data]
    
    return data

def process_screen_summary_android(summary_log):
    
    tv1 = summary_log[:,1]
    tv2 = summary_log[:,2]
    m = summary_log[:,3]
    u = summary_log[:,4]
    tv1e = summary_log[:,5]
    tv2e = summary_log[:,6]
    
    print('both tvs watched for (hr): %.2f, %.2f' % (tv1.sum()/3600.0, tv2.sum()/3600.0))
    #print('mobile used watched for (hr)', m.sum()/3600.0, u.sum()/3600.0)
    print('both tvs exposure (hr): %.2f, %.2f' % (tv1e.sum()/3600.0, tv2e.sum()/3600.0))
    
    tv1_m = tv1e*m # tv-1 exposure and mobile use (multitasking)
    tv2_m = tv2e*m # tv-2 exposure and mobile use (multitasking)
    
    tv1_u = tv1e*u # tv-1 exposure and mobile use with unidentified user (multitasking)
    tv2_u = tv2e*u # tv-2 exposure and mobile use with unidentified user (multitasking)

    # only tv viewing time - remove multitasking with tv viewinng from this
    tv1_ = tv1 - tv1*m - tv1*u 
    tv2_ = tv2 - tv2*m - tv2*u
    
    # only mobile use time - remove multitasking with tv exposure from this
    m_ = m - tv1_m - tv2_m
    u_ = u - tv1_u - tv2_u
    
    # multitasking - sum up multitasking across tv and mobile intersection
    multi = tv1_m + tv2_m + tv1_u + tv2_u 
    
    
    # only tv viewing time - remove tv viewing and multitasking from this
    tv1e_ = tv1e - tv1 - (tv1_m + tv1_u) + (tv1*m + tv1*u)
    tv2e_ = tv2e - tv2 - (tv2_m + tv2_u) + (tv2*m + tv2*u)
    
    
    return tv1_, tv2_, m_, u_, multi, tv1e_, tv2e_
    #tmp = np.zeros((summary_log.shape[0]))
    #return tmp, tmp, tmp, tmp, tmp, tmp, tv2e


def process_screen_summary_ios(summary_log):
    
    tv1 = summary_log[:,1]
    tv2 = summary_log[:,2]
    m = summary_log[:,3]
    u = summary_log[:,4]
    tv1e = summary_log[:,5]
    tv2e = summary_log[:,6]
    
    print('both tvs watched for (hr): %.2f, %.2f' % (tv1.sum()/3600.0, tv2.sum()/3600.0))
    #print('mobile used watched for (hr)', m.sum()/3600.0, u.sum()/3600.0)
    print('both tvs exposure (hr): %.2f, %.2f' % (tv1e.sum()/3600.0, tv2e.sum()/3600.0))
    

    # only tv viewing time - remove multitasking with tv viewinng from this
    tv1_ = tv1 #- tv1*m - tv1*u 
    tv2_ = tv2 #- tv2*m - tv2*u
    
    # only mobile use time - remove multitasking with tv exposure from this
    m_ = m #- tv1_m - tv2_m
    u_ = u #- tv1_u - tv2_u
    
    # multitasking - sum up multitasking across tv and mobile intersection
    # multi = tv1_m + tv2_m + tv1_u + tv2_u 
    
    
    # only tv viewing time - remove tv viewing and multitasking from this
    tv1e_ = tv1e - tv1 #- (tv1_m + tv1_u) + (tv1*m + tv1*u)
    tv2e_ = tv2e - tv2 #- (tv2_m + tv2_u) + (tv2*m + tv2*u)
    
    
    return tv1_, tv2_, m_, u_, tv1e_, tv2e_
    #tmp = np.zeros((summary_log.shape[0]))
    #return tmp, tmp, tmp, tmp, tmp, tmp, tv2e


def generate_viewlog(viewing_log, xl, xr):
    
    watch_time = []
    prsnt_time = []
    
    for ll in [0,6,12,18]:
        idx_filt = viewing_log[:,0]>=ll
        vl = viewing_log[idx_filt,:]
        
        idx_filt = vl[:,0]<ll+6
        vl = vl[idx_filt,:]
        
        watch_time.append(vl[:,1].sum()*5/60.0)
        prsnt_time.append(vl[:,2].sum()*5/60.0)
        
    
    watch_time = [round(f, 2) for f in watch_time]
    prsnt_time = [round(f, 2) for f in prsnt_time]
    
    print('watch TIME: ', watch_time)
    print('prsnt TIME: ', prsnt_time)
    
    return None

def generate_mobilelogs(fname, ios=False):
    data = open(fname,'r')
    data = data.readlines()
    
    data = [line.strip('\r\n').split(',') for line in data] # Other,gm,2023-02-09 07:00:11,2023-02-09 07:01:10,android.gm
    data = data[1:]
    user_logs = {'Child':{'duration': 0,'timestamps':[]},
                 'Other':{'duration': 0,'timestamps':[]},
                 'None': {'duration': 0,'timestamps':[]}
                }
    for d in data:
    
        #print(d)
        #time_start_obj = datetime.strptime(d[2], '%Y-%m-%d %H:%M:%S')
        #time_end_obj = datetime.strptime(d[3], '%Y-%m-%d %H:%M:%S')
        time_start_obj = datetime.strptime(d[6], '%H:%M:%S')
        time_end_obj = datetime.strptime(d[7], '%H:%M:%S')
        
        
        #dur_secs = (time_end_obj - time_start_obj).total_seconds()
        
        
        #time_start = convert_secs(d[2].split(' ')[1])
        #time_end = convert_secs(d[3].split(' ')[1])
        
        time_start = convert_secs(d[6])
        time_end = convert_secs(d[7])
        
        #print(time_start, time_end)
        #duration = time_end - time_start
        #print(duration, time_start, time_end)
        if time_end < time_start:
            print('echcek', time_start, time_end)
            print(d)
            time_end = 24*3600
        
        duration = (time_end-time_start)/3600.0
        user = d[5]
        if user=='Target Child' or user=='Target child':
            user = 'Child'
        elif user=='nan':
            user = 'None'
        
        if duration < 0:
            print(duration)
            print(d)
            
        user_logs[user]['duration'] = user_logs[user]['duration'] + duration
        user_logs[user]['timestamps'].append([time_start, time_end])
    
    return user_logs    

path = '/home/akv/FLASH_PO1/flash-mobile'
fnames3 = ['590_chronicle_android.csv', None, None]

android = True
#fname1 = '/home/akv/FLASH_PO1/data/581/'
#fname2 = '/home/akv/FLASH_PO1/data/581/581009_flashtv_data_04-12-2023_1fps_epc_tv.txt'

#fname3 = '/home/akv/FLASH_PO1/data/581/581_mobile_day1.csv'

#fig, ax = plt.subplots(3,1)

def plot_logs(ax, list_, labels): # [tv1, tv2, mobile, unident, multitasking, tv1e, tv2e]
    colors = {'tv1-gz':'tab:blue', 'tv2-gz':'tab:orange', 'mobile':'mediumseagreen', 
                'unident.':'k', 'multitask':'tab:purple', 'tv1-exp':'lightsteelblue', 'tv2-exp':'bisque'}
    
    for k, l in enumerate(list_):
        idxs = np.where(l==1)[0]
        if labels[k]=='unident.':
            continue
        ax.bar(idxs, 1.0, color=colors[labels[k]], width=1, label=labels[k])
        
    ax.set_ylim(0,1.0)
    ax.set_xlim(-2000,25*3600)
    ax.set_yticks([0,1.0])
    ax.set_xticks(ticks=np.arange(0,26*3600,2*3600),labels=np.arange(0,26,2))    
    

    return ax

fname3 = fnames3[0]
mobile_data = generate_mobilelogs(fname3, ios=False)
print(mobile_data[:10])

