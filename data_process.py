import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re


SET_CONFIG=4


def configset(num):
    config1 = np.append(np.full(36000,1,dtype=np.float),np.full(36000,0,dtype=np.float))
    config2 = np.full(72000,0.5,dtype=np.float)
    config3 = np.full(72000,0.8,dtype=np.float)
    config4 = np.concatenate((config1, config2, config3))
    switch={
        1:config1,
        2:config2,
        3:config3,
        4:config4
    }
    return switch.get(num,'Invalid Input')

def newest(DIR_PATH,num):
    switch={
        1:'cross.2phases_rou1_switch_rou0.xml',
        2:'cross.2phases_rou01_equal_300s.xml',
        3:'cross.2phases_rou01_unequal_5_300s.xml',
        4:'cross.all_synthetic.rou.xml'
    }
    search_string=switch.get(num,'Otherwise Invalid')

    files = os.listdir(DIR_PATH)
    FILE_LIST = [os.path.join(DIR_PATH, BASENAME) for BASENAME in files if bool(re.search(search_string,BASENAME))]
    return max(FILE_LIST, key=os.path.getctime)

def searchindx(starti,time,diff):
    indx=-1
    for i in range(starti,len(time)):
        if time[i]-time[starti]>=diff:
            indx=i
            break
    return indx
    
def timeavg(time,data,window):
    resultt = []
    resultd = []
    starti=0
    while(1):
        resultt.append(time[starti])
        endi=searchindx(starti,time,window)
        if endi>0:
            resultd.append(1-np.mean(data[starti:endi+1]))
            starti=endi+1
        else:
            resultd.append(1-np.mean(data[starti:-1]))
            break
    return resultt,resultd


PATH_TO_RECORDS = os.path.join("records", "one_run")
PATH_TO_CONFIG_RUN = newest(PATH_TO_RECORDS,SET_CONFIG)

time = []
current_phase = []
reward = []

PATH_TO_MEMORIES = os.path.join(PATH_TO_CONFIG_RUN,'memories.txt')
f1 = open(PATH_TO_MEMORIES,'r')
for row in f1:
    row = row.split()
    time.append(int(row[2]))
    current_phase.append(int(row[8]))
    reward.append(float(row[14]))


wndavg=2000
t0,rewardavg=timeavg(time,reward,wndavg)
plt.figure(1)    
plt.title("Rewards over time")
plt.xlabel('Time')
plt.ylabel('Reward')
plt.plot(t0,rewardavg, c='r')



cf_data = configset(SET_CONFIG)
t = np.arange(0,len(cf_data),1)

fig2, ax1 = plt.subplots()
color = '0'
ax1.set_xlabel('seconds')
ax1.set_ylabel('traffic ratio')
line1, = ax1.plot(t,cf_data,color=color,linestyle='--')
ax1.legend([line1],['Traffic Ratio'],loc='lower right')
ax2 = ax1.twinx()

t2,percstat=timeavg(time,current_phase,wndavg)

color = 'tab:orange'
ax2.set_ylabel("% of time for Green-WE")
line2, = ax2.plot(t2,percstat,color=color)
ax2.legend([line2],['Intellilight'],loc='upper right')

fig2.tight_layout()



count=[]
delay=[]
delay_var=[]
duration=[]
queue_length=[]
queue_length_var=[]
wait_time=[]
wait_time_var=[]

PATH_TO_LOGS = os.path.join(PATH_TO_CONFIG_RUN,'log_rewards.txt')

f2= open(PATH_TO_LOGS,'r')
first_line = f2.readline()
datalines = f2.readlines()
for line in datalines:
    
    line = line.split(',')
    if float(line[0])>700:
        count.append(float(line[0]))
        delay.append(float(line[2]))
        delay_var.append(float(line[3]))
        duration.append(float(line[4]))
        queue_length.append(float(line[12]))
        queue_length_var.append(float(line[13]))
        wait_time.append(float(line[14]))
        wait_time_var.append(float(line[15]))
    
fig3, ax3 = plt.subplots()
ax3.set_title('Queue Length')
ax3.set_xlabel('Count')
ax3.set_ylabel('Length (Variance)')
line3, =ax3.plot(count,duration,c='r')
# line4, =ax3.plot(count,queue_length_var,c='b')
# ax3.legend([line3,line4],['Queue Length','Queue Length Variance'])
df=pd.DataFrame({
    "Model Name": ["Intellilight"],
    "Reward": [np.mean(reward)],
    "Queue Length": [np.mean(queue_length)],
    "Delay": [np.mean(delay)],
    "Waiting Time": [np.mean(wait_time)],
    "Duration": [np.mean(duration)]
})
print(df)
plt.show()

