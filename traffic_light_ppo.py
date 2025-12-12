import copy
import json
import shutil

import os
import time
import math
import map_computor as map_computor
from deeplight_agent import DeeplightAgent
from ppo_agent import PPOAgent

from sumo_agent import SumoAgent
import xml.etree.ElementTree as ET


class TrafficLightPPO:
    """
    Docstring for TrafficLightPPO

    No Pretraining required for PPO
    """

    DIC_AGENTS = {
        # for PPO experiments map the default model name to the PPO agent
        "PPO": PPOAgent,
    }

    NO_PRETRAIN_AGENTS = []

    class ParaSet:

        def __init__(self, dic_paras):
            for key, value in dic_paras.items():
                setattr(self, key, value)

    class PathSet:

        # ======================================= conf files ========================================
        EXP_CONF = "exp.conf"
        SUMO_AGENT_CONF = "sumo_agent.conf"
        PATH_TO_CFG_TMP = os.path.join("data", "tmp")
        # ======================================= conf files ========================================

        # this is just for getting the file paths and sending the path output
        def __init__(self, path_to_conf, path_to_data, path_to_output, path_to_model):

            self.PATH_TO_CONF = path_to_conf
            self.PATH_TO_DATA = path_to_data
            self.PATH_TO_OUTPUT = path_to_output
            self.PATH_TO_MODEL = path_to_model

            if not os.path.exists(self.PATH_TO_OUTPUT):
                os.makedirs(self.PATH_TO_OUTPUT)
            if not os.path.exists(self.PATH_TO_MODEL):
                os.makedirs(self.PATH_TO_MODEL)

            dic_paras = json.load(
                open(os.path.join(self.PATH_TO_CONF, self.EXP_CONF), "r")
            )
            self.AGENT_CONF = "{0}_agent.conf".format(dic_paras["MODEL_NAME"].lower())
            self.TRAFFIC_FILE = dic_paras["TRAFFIC_FILE"]
            self.TRAFFIC_FILE_PRETRAIN = dic_paras["TRAFFIC_FILE_PRETRAIN"]

    def __init__(self, memo, f_prefix, epsilon):

        self.path_set = self.PathSet(
            os.path.join("conf", memo),
            os.path.join("data", memo),
            os.path.join("records", memo, f_prefix),
            os.path.join("model", memo, f_prefix),
        )

        self.para_set = self.load_conf(
            conf_file=os.path.join(self.path_set.PATH_TO_CONF, self.path_set.EXP_CONF)
        )
        shutil.copy(
            os.path.join(self.path_set.PATH_TO_CONF, self.path_set.EXP_CONF),
            os.path.join(self.path_set.PATH_TO_OUTPUT, self.path_set.EXP_CONF),
        )

        self.agent = self.DIC_AGENTS[self.para_set.MODEL_NAME](
            num_phases=2, num_actions=2, path_set=self.path_set
        )

        self.epsilon = epsilon

    def load_conf(self, conf_file):

        dic_paras = json.load(open(conf_file, "r"))
        return self.ParaSet(dic_paras)

    @staticmethod
    def _set_traffic_file(
        sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name
    ):

        # update sumocfg
        sumo_cfg = ET.parse(sumo_config_file_tmp_name)
        config_node = sumo_cfg.getroot()
        input_node = config_node.find("input")
        for route_files in input_node.findall("route-files"):
            input_node.remove(route_files)
        input_node.append(
            ET.Element(
                "route-files", attrib={"value": ",".join(list_traffic_file_name)}
            )
        )
        sumo_cfg.write(sumo_config_file_output_name)

    def set_traffic_file(self):

        self._set_traffic_file(
            os.path.join(self.path_set.PATH_TO_DATA, "cross.sumocfg"),
            os.path.join(self.path_set.PATH_TO_DATA, "cross.sumocfg"),
            self.para_set.TRAFFIC_FILE,
        )
        for file_name in self.path_set.TRAFFIC_FILE:
            shutil.copy(
                os.path.join(self.path_set.PATH_TO_DATA, file_name),
                os.path.join(self.path_set.PATH_TO_OUTPUT, file_name),
            )

    def run(self, sumo_cmd_str, use_average):
        """
        PPO-Clip
        """
        total_run_cnt = self.para_set.RUN_COUNTS

        file_name_memory = os.path.join(self.path_set.PATH_TO_OUTPUT, "memories.txt")

        # start sumo
        s_agent = SumoAgent(sumo_cmd_str, self.path_set)
        current_time = s_agent.get_current_time()  # in seconds

        # action = 0 turns the lights yellow
        # action = 1 flips the lights

        last_update_time = 0

        while current_time < total_run_cnt:

            f_memory = open(file_name_memory, "a")

            # get state
            state = s_agent.get_observation()
            state = self.agent.get_state(state, current_time)
            # choose action from policy
            action_pred, action_probs = self.agent.choose(
                count=current_time, if_pretrain=False
            )

            # perform action in SUMO
            reward, action = s_agent.take_action(action_pred)

            # get next state
            next_state = s_agent.get_observation()
            next_state = self.agent.get_next_state(next_state, current_time)

            # remember transition (on-policy)
            self.agent.remember(state, action, reward, next_state)

            # log
            memory_str = (
                "time = %d\taction = %d\tcurrent_phase = %d\tnext_phase = %d\treward = %f"
                % (
                    current_time,
                    action,
                    state.cur_phase[0][0],
                    state.next_phase[0][0],
                    reward,
                )
            )
            print(memory_str)
            f_memory.write(memory_str + "\n")
            f_memory.close()

            current_time = s_agent.get_current_time()  # in seconds

            # periodically update PPO (on-policy)
            if hasattr(self.para_set, "UPDATE_PERIOD"):
                update_period = self.para_set.UPDATE_PERIOD
            else:
                update_period = 300

            if current_time - last_update_time >= update_period:
                batch_size = len(self.agent.episode_memory)
                print(f"[PPO UPDATE] time={current_time}, batch_size={batch_size}")
                self.agent.update_network(
                    if_pretrain=False,
                    use_average=use_average,
                    current_time=current_time,
                )
                self.agent.save_model(f"ckpt_{int(current_time)}")
                print(f"[PPO SAVED] checkpoint saved: ckpt_{int(current_time)}")
                last_update_time = current_time

        # final update if any remaining transitions
        if len(self.agent.episode_memory) > 0:
            print(
                f"[PPO FINAL UPDATE] time={current_time}, batch_size={len(self.agent.episode_memory)}"
            )
            self.agent.update_network(
                if_pretrain=False, use_average=use_average, current_time=current_time
            )
            self.agent.save_model(f"final_ckpt_{int(current_time)}")
            print(f"[PPO SAVED] final checkpoint saved: final_ckpt_{int(current_time)}")

    @staticmethod
    def main(memo, f_prefix, sumo_cmd_str, sumo_cmd_pretrain_str=None, epsilon=0.1):
        player = TrafficLightPPO(memo, f_prefix, epsilon)
        player.set_traffic_file()
        # PPO is on-policy, so no pretrain phase in this repo's default design
        player.run(sumo_cmd_str, use_average=False)
