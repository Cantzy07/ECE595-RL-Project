# -*- coding: utf-8 -*-

'''
@author: hzw77, gjz5038

python TrafficLightDQN.py SEED setting_memo

SEED: random number for initializing the experiment
setting_memo: the folder name for this experiment
    The conf, data files will should be placed in conf/setting_memo, data/setting_memo respectively
    The records, model files will be generated in records/setting_memo, model/setting_memo respectively

'''


import copy
import json
import shutil

import os
import time
import math
import map_computor as map_computor

from sumo_agent import SumoAgent
import xml.etree.ElementTree as ET
import ppo2

import random
import numpy as np
import os.path as osp
from collections import deque
import tensorflow as tf

from runner import Runner
from ppo_model import Model

def constfn(val):
    def f(_):
        return val
    return f

# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

class SimpleLogger:
    def __init__(self, logdir=None):
        self.logdir = logdir
        self.kvs = {}

    def set_dir(self, logdir):
        self.logdir = logdir
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)

    def get_dir(self):
        return self.logdir

    def info(self, msg):
        print(msg)

    def logkv(self, key, value):
        self.kvs[key] = value

    def dumpkvs(self):
        if not self.kvs:
            return
        msg = " | ".join(f"{k}: {v}" for k, v in self.kvs.items())
        print(msg)
        self.kvs = {}

logger = SimpleLogger()

def set_global_seeds(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def explained_variance(y_pred, y_true):
    var_y = np.var(y_true)
    if var_y == 0:
        return np.nan
    return 1 - np.var(y_true - y_pred) / (var_y + 1e-8)

class TrafficLightDQN:

    DIC_AGENTS = {
        "Deeplight": ppo2,
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

        def __init__(self, path_to_conf, path_to_data, path_to_output, path_to_model):

            self.PATH_TO_CONF = path_to_conf
            self.PATH_TO_DATA = path_to_data
            self.PATH_TO_OUTPUT = path_to_output
            self.PATH_TO_MODEL = path_to_model

            if not os.path.exists(self.PATH_TO_OUTPUT):
                os.makedirs(self.PATH_TO_OUTPUT)
            if not os.path.exists(self.PATH_TO_MODEL):
                os.makedirs(self.PATH_TO_MODEL)

            dic_paras = json.load(open(os.path.join(self.PATH_TO_CONF, self.EXP_CONF), "r"))
            self.AGENT_CONF = "{0}_agent.conf".format(dic_paras["MODEL_NAME"].lower())
            self.TRAFFIC_FILE = dic_paras["TRAFFIC_FILE"]
            self.TRAFFIC_FILE_PRETRAIN = dic_paras["TRAFFIC_FILE_PRETRAIN"]

    def __init__(self, memo, f_prefix):

        self.path_set = self.PathSet(os.path.join("conf", memo),
                                     os.path.join("data", memo),
                                     os.path.join("records", memo, f_prefix),
                                     os.path.join("model", memo, f_prefix))

        self.para_set = self.load_conf(conf_file=os.path.join(self.path_set.PATH_TO_CONF, self.path_set.EXP_CONF))
        shutil.copy(
            os.path.join(self.path_set.PATH_TO_CONF, self.path_set.EXP_CONF),
            os.path.join(self.path_set.PATH_TO_OUTPUT, self.path_set.EXP_CONF))

        self.agent = self.DIC_AGENTS[self.para_set.MODEL_NAME](num_phases=2,
                                                               num_actions=2,
                                                               path_set=self.path_set)

    def load_conf(self, conf_file):

        dic_paras = json.load(open(conf_file, "r"))
        return self.ParaSet(dic_paras)

    def check_if_need_pretrain(self):

        if self.para_set.MODEL_NAME in self.NO_PRETRAIN_AGENTS:
            return False
        else:
            return True

    def _generate_pre_train_ratios(self, phase_min_time, em_phase):
        phase_traffic_ratios = [phase_min_time]

        # generate how many varients for each phase
        for i, phase_time in enumerate(phase_min_time):
            if i == em_phase:
                for j in range(1, 5, 1):
                    gen_phase_time = copy.deepcopy(phase_min_time)
                    gen_phase_time[i] += j
                    phase_traffic_ratios.append(gen_phase_time)
            else:
                # pass
                for j in range(1, 5, 1):
                    gen_phase_time = copy.deepcopy(phase_min_time)
                    gen_phase_time[i] += j
                    phase_traffic_ratios.append(gen_phase_time)
            for j in range(5, 20, 5):
                gen_phase_time = copy.deepcopy(phase_min_time)
                gen_phase_time[i] += j
                phase_traffic_ratios.append(gen_phase_time)

        return phase_traffic_ratios

    @staticmethod
    def _set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):

        # update sumocfg
        sumo_cfg = ET.parse(sumo_config_file_tmp_name)
        config_node = sumo_cfg.getroot()
        input_node = config_node.find("input")
        for route_files in input_node.findall("route-files"):
            input_node.remove(route_files)
        input_node.append(
            ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
        sumo_cfg.write(sumo_config_file_output_name)

    def set_traffic_file(self):

        self._set_traffic_file(
            os.path.join(self.path_set.PATH_TO_DATA, "cross_pretrain.sumocfg"),
            os.path.join(self.path_set.PATH_TO_DATA, "cross_pretrain.sumocfg"),
            self.para_set.TRAFFIC_FILE_PRETRAIN)
        self._set_traffic_file(
            os.path.join(self.path_set.PATH_TO_DATA, "cross.sumocfg"),
            os.path.join(self.path_set.PATH_TO_DATA, "cross.sumocfg"),
            self.para_set.TRAFFIC_FILE)
        for file_name in self.path_set.TRAFFIC_FILE_PRETRAIN:
            shutil.copy(
                    os.path.join(self.path_set.PATH_TO_DATA, file_name),
                    os.path.join(self.path_set.PATH_TO_OUTPUT, file_name))
        for file_name in self.path_set.TRAFFIC_FILE:
            shutil.copy(
                os.path.join(self.path_set.PATH_TO_DATA, file_name),
                os.path.join(self.path_set.PATH_TO_OUTPUT, file_name))

    def learn(*, network, env, total_timesteps, eval_env=None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, log_dir=None, **network_kwargs):
        '''
        Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

        Parameters:
        ----------

        network: callable that builds a policy when invoked as
                ``network(ob_space, ac_space, nbatch, nsteps, sess, **network_kwargs)``

        env: vectorized environment exposing the OpenAI Gym VecEnv API.


        nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                        nenv is number of environment copies simulated in parallel)

        total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

        ent_coef: float                   policy entropy coefficient in the optimization objective

        lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                        training and 0 is the end of the training.

        vf_coef: float                    value function loss coefficient in the optimization objective

        max_grad_norm: float or None      gradient norm clipping coefficient

        gamma: float                      discounting factor

        lam: float                        advantage estimation discounting factor (lambda in the paper)

        log_interval: int                 number of timesteps between logging events

        nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                        should be smaller or equal than number of environments run in parallel.

        noptepochs: int                   number of training epochs per update

        cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                        and 0 is the end of the training

        save_interval: int                number of timesteps between saving events

        load_path: str                    path to load the model from

        **network_kwargs:                 keyword arguments forwarded to the policy builder.



        '''

        set_global_seeds(seed)

        if isinstance(lr, float): lr = constfn(lr)
        else: assert callable(lr)
        if isinstance(cliprange, float): cliprange = constfn(cliprange)
        else: assert callable(cliprange)
        total_timesteps = int(total_timesteps)

        if log_dir is not None:
            logger.set_dir(log_dir)

        if not callable(network):
            raise ValueError("network must be a callable policy builder")

        def policy(nbatch, nsteps, sess):
            return network(env.observation_space, env.action_space, nbatch, nsteps, sess, **network_kwargs)

        # Get the nb of env
        nenvs = env.num_envs

        # Get state_space and action_space
        ob_space = env.observation_space
        ac_space = env.action_space

        # Calculate the batch_size
        nbatch = nenvs * nsteps
        nbatch_train = nbatch // nminibatches
        is_mpi_root = True

        if model_fn is None:
            model_fn = Model

        model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                        nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm)

        if load_path is not None:
            model.load(load_path)
        # Instantiate the runner object
        runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
        if eval_env is not None:
            eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)

        epinfobuf = deque(maxlen=100)
        if eval_env is not None:
            eval_epinfobuf = deque(maxlen=100)

        if init_fn is not None:
            init_fn()

        # Start total timer
        tfirststart = time.perf_counter()

        nupdates = total_timesteps//nbatch
        for update in range(1, nupdates+1):
            assert nbatch % nminibatches == 0
            # Start timer
            tstart = time.perf_counter()
            frac = 1.0 - (update - 1.0) / nupdates
            # Calculate the learning rate
            lrnow = lr(frac)
            # Calculate the cliprange
            cliprangenow = cliprange(frac)

            if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

            # Get minibatch
            obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
            if eval_env is not None:
                eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

            if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

            epinfobuf.extend(epinfos)
            if eval_env is not None:
                eval_epinfobuf.extend(eval_epinfos)

            # Here what we're going to do is for each minibatch calculate the loss and append it.
            mblossvals = []
            if states is None: # nonrecurrent version
                # Index of each element of batch_size
                # Create the indices array
                inds = np.arange(nbatch)
                for _ in range(noptepochs):
                    # Randomize the indexes
                    np.random.shuffle(inds)
                    # 0 to batch_size with batch_train_size step
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices))
            else: # recurrent version
                assert nenvs % nminibatches == 0
                envsperbatch = nenvs // nminibatches
                envinds = np.arange(nenvs)
                flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
                for _ in range(noptepochs):
                    np.random.shuffle(envinds)
                    for start in range(0, nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()
                        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mbstates = states[mbenvinds]
                        mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

            # Feedforward --> get losses --> update
            lossvals = np.mean(mblossvals, axis=0)
            # End timer
            tnow = time.perf_counter()
            # Calculate the fps (frame per second)
            fps = int(nbatch / (tnow - tstart))

            if update_fn is not None:
                update_fn(update)

            if update % log_interval == 0 or update == 1:
                # Calculates if value function is a good predicator of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
                ev = explained_variance(values, returns)
                logger.logkv("misc/serial_timesteps", update*nsteps)
                logger.logkv("misc/nupdates", update)
                logger.logkv("misc/total_timesteps", update*nbatch)
                logger.logkv("fps", fps)
                logger.logkv("misc/explained_variance", float(ev))
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
                if eval_env is not None:
                    logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                    logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
                logger.logkv('misc/time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    logger.logkv('loss/' + lossname, lossval)

                logger.dumpkvs()
            if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
                checkdir = osp.join(logger.get_dir(), 'checkpoints')
                os.makedirs(checkdir, exist_ok=True)
                savepath = osp.join(checkdir, '%.5i'%update)
                print('Saving to', savepath)
                model.save(savepath)

        return model


def main(memo, f_prefix, sumo_cmd_str, sumo_cmd_pretrain_str):

    player = TrafficLightDQN(memo, f_prefix)
    player.set_traffic_file()
    player.learn(sumo_cmd_pretrain_str, if_pretrain=True, use_average=True)
    player.learn(sumo_cmd_str, if_pretrain=False, use_average=False)

