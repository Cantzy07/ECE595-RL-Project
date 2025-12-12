import time
import os
import sys
import json
import time
from traffic_light_ppo import TrafficLightPPO

# Use the existing config folder 'one_run'
memo = "one_run"
# create a prefix similar to runexp.py
SEED = 31200
prefix = (
    "PPO_"
    + time.strftime("%m_%d_%H_%M_%S_", time.localtime(time.time()))
    + "seed_%d" % SEED
)

# construct sumo command structure; map_computor.start_sumo will use the last element as the config path
base_dir = os.path.split(os.path.realpath(__file__))[0]
sumo_cfg = os.path.join(base_dir, "data", memo, "cross.sumocfg")
sumoCmd = ["sumo", "-c", sumo_cfg]

print("Starting PPO experiment with config:", sumo_cfg)

player = TrafficLightPPO(memo, prefix, epsilon=0.1)
player.set_traffic_file()
# run (this will start SUMO via TraCI; it may open SUMO GUI and run for the duration in conf/one_run/exp.conf)
player.run(sumoCmd, use_average=False)

print("PPO experiment finished")
