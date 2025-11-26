import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

time = []
reward = []

f1 = open("records/one_run/Deeplight_['cross.2phases_rou1_switch_rou0.xml']_['cross.2phases_rou1_switch_rou0.xml']_11_25_21_49_30_seed_31200/memories.txt",'r')
for row in f1:
    row = row.split()
    time.append(int(row[2]))
    reward.append(float(row[14]))
    
plt.title("Rewards over time")
plt.xlabel('Time')
plt.ylabel('Reward')
plt.plot(time,reward, c='r')

plt.show()

count=[]
queue_length=[]
queue_length_var=[]

f2= open("records/one_run/Deeplight_['cross.2phases_rou1_switch_rou0.xml']_['cross.2phases_rou1_switch_rou0.xml']_11_25_21_49_30_seed_31200/log_rewards.txt",'r')
first_line = f2.readline()
datalines = f2.readlines()
for line in datalines:
    
    line = line.split(',')
    count.append(float(line[0]))
    queue_length.append(float(line[13]))
    queue_length_var.append(float(line[14]))
    
df = pd.DataFrame(queue_length_var,count)
df_mva=df.rolling(100).mean()
df_mva=df_mva.dropna()
df_mva.plot(legend=False)
plt.title("Queue Length Variance")
plt.xlabel('Count')
plt.ylabel('Variance')
plt.plot(count,queue_length_var,c='r')

plt.show()
