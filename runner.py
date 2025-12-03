import numpy as np


class Runner:
    """
    Collects rollouts from a vectorized environment and computes
    generalized advantage estimates for PPO updates.
    """

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        self.nsteps = nsteps
        self.gamma = gamma
        self.lam = lam
        self.obs = np.asarray(env.reset())
        self.dones = np.zeros(env.num_envs, dtype=np.bool_)
        self.states = model.initial_state

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones.copy())
            next_obs, rewards, dones, infos = self.env.step(actions)
            self.obs = np.asarray(next_obs)
            self.dones = np.asarray(dones, dtype=np.bool_)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool_)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones.astype(np.float32)
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1].astype(np.float32)
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            mb_advs[t] = lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
