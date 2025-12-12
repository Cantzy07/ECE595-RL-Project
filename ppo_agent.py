import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import random

from network_agent import NetworkAgent, State
from agent import Agent


class PPOAgent(NetworkAgent):
    """A minimal PPO-Clip agent for discrete actions.

    This agent is intentionally lightweight so it can integrate with the
    repository's `Agent` / `NetworkAgent` interfaces. It implements the
    methods used by `traffic_light_ppo.py`.
    """

    def __init__(self, num_phases, num_actions, path_set):
        super(PPOAgent, self).__init__(num_phases, path_set)

        self.num_actions = num_actions

        # hyper-parameters (fallbacks if not present in agent conf)
        self.clip_eps = getattr(self.para_set, "CLIP_EPS", 0.2)
        self.ppo_epochs = getattr(self.para_set, "PPO_EPOCHS", 10)
        self.batch_size = getattr(self.para_set, "PPO_BATCH_SIZE", 64)
        self.gamma = getattr(self.para_set, "GAMMA", 0.99)
        self.entropy_coef = getattr(self.para_set, "ENTROPY_COEF", 0.01)

        lr = getattr(self.para_set, "LEARNING_RATE", 3e-4)

        # build simple actor and critic networks based on the repo's feature scheme
        self._build_networks(lr)

        # storage for one-on-policy batch
        self.episode_memory = []

    def _build_inputs(self):
        inputs = {}
        for feature_name in self.para_set.LIST_STATE_FEATURE:
            shape = getattr(State, "D_" + feature_name.upper())
            inputs[feature_name] = Input(shape=shape, name="input_" + feature_name)
        return inputs

    def _build_flatten(self, inputs):
        # flatten / concatenate features (reuse NetworkAgent helpers for CNN if necessary)
        flattened = []
        for feature_name in self.para_set.LIST_STATE_FEATURE:
            if len(getattr(State, "D_" + feature_name.upper())) > 1:
                x = self._cnn_network_structure(inputs[feature_name])
            else:
                x = Flatten()(inputs[feature_name])
            flattened.append(x)
        if len(flattened) > 1:
            all_feat = concatenate(flattened, axis=1)
        else:
            all_feat = flattened[0]
        shared = self._shared_network_structure(
            all_feat, getattr(self.para_set, "D_DENSE", 32)
        )
        return inputs, shared

    def _build_networks(self, lr):
        inputs = self._build_inputs()
        inputs, shared = self._build_flatten(inputs)

        # actor head
        a_hidden = Dense(64, activation="tanh")(shared)
        a_logits = Dense(self.num_actions, activation="softmax", name="actor_out")(
            a_hidden
        )

        # critic head
        c_hidden = Dense(64, activation="tanh")(shared)
        c_value = Dense(1, activation="linear", name="critic_out")(c_hidden)

        self.actor = Model(inputs=list(inputs.values()), outputs=a_logits)
        self.critic = Model(inputs=list(inputs.values()), outputs=c_value)

        self.actor_optimizer = Adam(learning_rate=lr)
        self.critic_optimizer = Adam(learning_rate=lr)

    def convert_state_to_input(self, state):
        return [
            getattr(state, feature_name)
            for feature_name in self.para_set.LIST_STATE_FEATURE
        ]

    def choose(self, count, if_pretrain=False):
        """Sample an action from the policy for current state."""
        probs = self.actor.predict(self.convert_state_to_input(self.state),batch_size=self.batch_size)
        if if_pretrain:
            action = np.argmax(probs[0])
        else:
            if random.random() <= self.para_set.EPSILON:  # continue explore new Random Action
                action = random.randrange(len(probs[0]))
                print("##Explore")
            else:  # exploitation
                action = np.argmax(probs[0])
            if self.para_set.EPSILON > 0.001 and count >= 20000:
                self.para_set.EPSILON = self.para_set.EPSILON * 0.9999
        return action, probs

    def remember(self, state, action, reward, next_state):
        # store tuple; log-probs and values will be computed at update time
        self.episode_memory.append(
            (state, action, reward, next_state, state.if_terminal)
        )

    def _prepare_batch(self):
        # Convert episode memory into arrays for training
        states = [m[0] for m in self.episode_memory]
        actions = np.array([m[1] for m in self.episode_memory])
        rewards = np.array([m[2] for m in self.episode_memory])
        dones = np.array([m[4] for m in self.episode_memory]).astype(np.float32)
        
        Xs = []
        for feature_name in self.para_set.LIST_STATE_FEATURE:
            Xs.append(np.vstack([getattr(s,feature_name) for s in states]))

        # values and old_probs
        values = self.critic.predict(Xs,batch_size=self.batch_size).flatten()
        old_probs = self.actor.predict(Xs,batch_size=self.batch_size)

        # compute returns
        returns = np.zeros_like(rewards, dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running * (1.0 - dones[t])
            returns[t] = running

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return Xs, actions, old_probs, returns, advantages

    def update_network(self, if_pretrain, use_average, current_time):
        if len(self.episode_memory) == 0:
            return

        Xs, actions, old_probs, returns, advantages = self._prepare_batch()

        dataset_size = len(actions)

        for epoch in range(self.ppo_epochs):
            # simple full-batch or minibatch loop
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_idx = idxs[start:end]

                mb_Xs = [x[mb_idx] for x in Xs]
                mb_actions = actions[mb_idx]
                mb_old_probs = old_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # train critic
                with tf.GradientTape() as tape_c:
                    values_pred = tf.squeeze(self.critic(mb_Xs, training=True), axis=1)
                    critic_loss = tf.reduce_mean(tf.square(mb_returns - values_pred))
                grads_c = tape_c.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(
                    zip(grads_c, self.critic.trainable_variables)
                )

                # train actor with clipped surrogate objective
                with tf.GradientTape() as tape_a:
                    probs = self.actor(mb_Xs, training=True)
                    action_probs = tf.reduce_sum(
                        probs * tf.one_hot(mb_actions, self.num_actions), axis=1
                    )
                    old_action_probs = tf.reduce_sum(
                        tf.constant(mb_old_probs, dtype=tf.float32)
                        * tf.one_hot(mb_actions, self.num_actions),
                        axis=1,
                    )
                    ratio = action_probs / (old_action_probs + 1e-8)
                    clipped = tf.clip_by_value(
                        ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                    )
                    actor_loss = -tf.reduce_mean(
                        tf.minimum(ratio*mb_advantages, clipped*mb_advantages)
                    )
                    entropy = -tf.reduce_mean(
                        tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
                    )
                    total_loss = actor_loss - self.entropy_coef * entropy
                grads_a = tape_a.gradient(total_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(
                    zip(grads_a, self.actor.trainable_variables)
                )

        # clear memory after update (on-policy)
        self.episode_memory = []

    def save_model(self, prefix):
        # save actor and critic
        self.actor.save(os.path.join(self.path_set.PATH_TO_MODEL, f"{prefix}_actor.h5"))
        self.critic.save(
            os.path.join(self.path_set.PATH_TO_MODEL, f"{prefix}_critic.h5")
        )
