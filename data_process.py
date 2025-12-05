import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

time = []
reward = []

f1 = open(
    "records/one_run/PPO_12_04_02_06_13_seed_31200/memories.txt",
    "r",
)
for row in f1:
    row = row.split()
    time.append(int(row[2]))
    reward.append(float(row[14]))

plt.title("Rewards over time")
plt.xlabel("Time")
plt.ylabel("Reward")
plt.plot(time, reward, c="r")

plt.show()

count = []
queue_length = []
queue_length_var = []

f2 = open(
    "records/one_run/PPO_12_04_02_06_13_seed_31200/log_rewards.txt",
    "r",
)
first_line = f2.readline()
datalines = f2.readlines()
for line in datalines:

    line = line.split(",")
    count.append(float(line[0]))
    queue_length.append(float(line[13]))
    queue_length_var.append(float(line[14]))

df = pd.DataFrame(queue_length_var, index=count)

fig, ax = plt.subplots()
df_mva = df.rolling(100).mean().dropna()
df_mva.plot(ax=ax, label="Queue Length (Moving Window Average)")


ax.plot(count, queue_length_var, label="Queue Length Variance", color="r")
ax.set_title("Queue Length Variance")
ax.set_xlabel("Count")
ax.set_ylabel("Variance")

ax.legend()

plt.show()
