import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

time = []
reward = []

f1 = open("records/one_run/Deeplight_['cross.2phases_rou1_switch_rou0.xml']_['cross.2phases_rou1_switch_rou0.xml']_12_08_12_29_49_seed_31200/memories.txt",'r')
for row in f1:
    row = row.split()
    if int(row[2])>672:
        time.append(int(row[2]))
        reward.append(float(row[14]))
    
plt.title("Rewards over time")
plt.xlabel('Time')
plt.ylabel('Reward')
plt.plot(time,reward, c='r')

plt.show()

count=[]
delay=[]
delay_var=[]
queue_length=[]
queue_length_var=[]
wait_time=[]
wait_time_var=[]

f2= open("records/one_run/Deeplight_['cross.2phases_rou1_switch_rou0.xml']_['cross.2phases_rou1_switch_rou0.xml']_12_08_12_29_49_seed_31200/log_rewards.txt",'r')
first_line = f2.readline()
datalines = f2.readlines()
for line in datalines:
    
    line = line.split(',')
    if float(line[0])>700:
        count.append(float(line[0]))
        delay.append(float(line[3]))
        delay_var.append(float(line[4]))
        queue_length.append(float(line[12]))
        queue_length_var.append(float(line[13]))
        wait_time.append(float(line[14]))
        wait_time_var.append(float(line[15]))
    
plt.title("Queue Length (Variance)")
plt.xlabel('Count')
plt.ylabel('Length (Variance)')
plt.plot(count,queue_length,c='r')
plt.plot(count,queue_length_var,c='b')

plt.show()
print("%d",np.mean(queue_length))
