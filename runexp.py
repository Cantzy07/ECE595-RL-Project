# -*- coding: utf-8 -*-

"""
@author: hzw77, gjz5038

python runexp.py

Run experiments in batch with configuration

"""

# ================================= only change these sections ==========================================
SEED = 31200

setting_memo = "one_run"

# Set algorithm: "DQN" or "PPO"
ALGORITHM = "PPO"  # Change to "DQN" to run DQN experiments


# first column: for train, second column: for spre_train
list_traffic_files = [
    [["cross.2phases_rou1_switch_rou0.xml"], ["cross.2phases_rou1_switch_rou0.xml"]],
    [["cross.2phases_rou01_equal_300s.xml"], ["cross.2phases_rou01_equal_300s.xml"]],
    [
        ["cross.2phases_rou01_unequal_5_300s.xml"],
        ["cross.2phases_rou01_unequal_5_300s.xml"],
    ],
    [["cross.all_synthetic.rou.xml"], ["cross.all_synthetic.rou.xml"]],
]

list_model_name = [
    "Deeplight",
]

# ================================= only change these sections ==========================================


import random

random.seed(SEED)
import numpy as np
import tensorflow as tf

np.random.seed(SEED)
# Below is deprecated in later versions of tf
# from tensorflow import set_random_seed
# set_random_seed((SEED))
tf.random.set_seed((SEED))
import json
import os
import sys
import traffic_light_dqn
import traffic_light_ppo
import time

PATH_TO_CONF = os.path.join("conf", setting_memo)

sumoBinary = os.path.join(sys.prefix, "bin/sumo-gui")
sumoCmd = [
    sumoBinary,
    "-c",
    r"{0}/data/{1}/cross.sumocfg".format(
        os.path.split(os.path.realpath(__file__))[0], setting_memo
    ),
]
sumoCmd_pretrain = [
    sumoBinary,
    "-c",
    r"{0}/data/{1}/cross_pretrain.sumocfg".format(
        os.path.split(os.path.realpath(__file__))[0], setting_memo
    ),
]

sumoBinary_nogui = os.path.join(sys.prefix, "bin/sumo")
sumoCmd_nogui = [
    sumoBinary_nogui,
    "-c",
    r"{0}/data/{1}/cross.sumocfg".format(
        os.path.split(os.path.realpath(__file__))[0], setting_memo
    ),
]
sumoCmd_nogui_pretrain = [
    sumoBinary_nogui,
    "-c",
    r"{0}/data/{1}/cross_pretrain.sumocfg".format(
        os.path.split(os.path.realpath(__file__))[0], setting_memo
    ),
]

for model_name in list_model_name:
    for traffic_file, traffic_file_pretrain in list_traffic_files:
        dic_exp = json.load(open(os.path.join(PATH_TO_CONF, "exp.conf"), "r"))
        dic_exp["MODEL_NAME"] = model_name
        dic_exp["TRAFFIC_FILE"] = traffic_file
        dic_exp["TRAFFIC_FILE_PRETRAIN"] = traffic_file_pretrain
        if "real" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 86400
        elif "2phase" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 72000
        elif "synthetic" in traffic_file[0]:
            dic_exp["RUN_COUNTS"] = 216000

        if "real" in traffic_file[0]:
            dic_exp["RUN_COUNTS_PRETRAINS"] = 86400
        elif "2phase" in traffic_file[0]:
            dic_exp["RUN_COUNTS_PRETRAIN"] = 36000
        elif "synthetic" in traffic_file[0]:
            dic_exp["RUN_COUNTS_PRETRAIN"] = 103500
        json.dump(dic_exp, open(os.path.join(PATH_TO_CONF, "exp.conf"), "w"), indent=4)

        # change MIN_ACTION_TIME correspondingly

        dic_sumo = json.load(open(os.path.join(PATH_TO_CONF, "sumo_agent.conf"), "r"))
        if model_name == "Deeplight":
            dic_sumo["MIN_ACTION_TIME"] = 5
        else:
            dic_sumo["MIN_ACTION_TIME"] = 1
        json.dump(
            dic_sumo, open(os.path.join(PATH_TO_CONF, "sumo_agent.conf"), "w"), indent=4
        )

        prefix = "{0}_{1}_{2}_{3}".format(
            dic_exp["MODEL_NAME"],
            dic_exp["TRAFFIC_FILE"],
            dic_exp["TRAFFIC_FILE_PRETRAIN"],
            time.strftime("%m_%d_%H_%M_%S_", time.localtime(time.time()))
            + "seed_%d" % SEED,
        )
        if model_name == "PPO":
            print(f"[PPO RUN] Starting PPO experiment with {traffic_file}")
            traffic_light_ppo.TrafficLightPPO.main(
                memo=setting_memo,
                f_prefix=prefix,
                sumo_cmd_str=sumoCmd_nogui,
                epsilon=0.1,
            )
        elif model_name == "Deeplight":  # DQN
            print(f"[DQN RUN] Starting DQN experiment with {traffic_file}")
            traffic_light_dqn.main(
                memo=setting_memo,
                f_prefix=prefix,
                sumo_cmd_str=sumoCmd_nogui,
                sumo_cmd_pretrain_str=sumoCmd_nogui_pretrain,
            )
        print("finished {0}".format(traffic_file))
    print("finished {0}".format(model_name))
