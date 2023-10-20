import gymnasium as gym
from gymnasium.wrappers import TransformReward
from gymnasium import spaces
from minigrid.wrappers import *
import numpy as np

class FlattenWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        imgSpace = env.observation_space.spaces['image']
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=imgSpace.shape,
            dtype='uint8'
        )
    #
    # def step(self, action):
    #     obs, reward, done, info = self.env.step(action)
    #
    #     env = self.unwrapped
    #     tup = (tuple(env.agent_pos), env.agent_dir, action)
    #
    #     # Get the count for this (s,a) pair
    #     pre_count = 0
    #     if tup in self.counts:
    #         pre_count = self.counts[tup]
    #
    #     # Update the count for this (s,a) pair
    #     new_count = pre_count + 1
    #     self.counts[tup] = new_count
    #
    #     bonus = 1 / math.sqrt(new_count)
    #     reward += bonus
    #
    #     return obs, reward, done, info
    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return full_grid


class ReturnWrapper(gym.Wrapper):
    #######################################################################
    # Copyright (C) 2020 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
    # Permission given to modify the code as long as you keep this        #
    # declaration at the top                                              #
    #######################################################################
    def __init__(self, env):
        super().__init__(env)
        self.total_rewards = 0
        self.steps = 0

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward=reward*100-self.steps-1

        # if np.any([done, truncated]):
        #     done = True

        if done or truncated:
            info['returns/episodic_reward'] = self.total_rewards
            info['returns/episodic_length'] = self.steps
            self.total_rewards = 0
            self.steps = 0
        else:
            info['returns/episodic_reward'] = None
            info['returns/episodic_length'] = None

        return obs, reward, done, truncated, info

class ReturnWrapper_wargs(ReturnWrapper):
    #######################################################################
    # Copyright (C) 2020 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
    # Permission given to modify the code as long as you keep this        #
    # declaration at the top                                              #
    #######################################################################
    def __init__(self, env, reward_reg=5000, max_steps=1000):
        super().__init__(env)
        self.total_rewards = 0
        self.steps = 0
        self.multiplier = reward_reg
        self.max_steps = max_steps
        self.step_discount = 1

    def step(self, action):
        # eps = 0.25
        # if np.random.rand() < eps:
        #     action = self.env.action_space.sample()

        obs, reward, done, truncated, info = self.env.step(action)
        # reward = np.ceil(reward)*self.multiplier/(self.steps+1)
        # reward -= self.step_discount
        # print reward if reward is not 0
        if reward != 0:
            reward = 1
        reward = reward * self.multiplier - 1
        # reward = reward * self.multiplier - 1
        self.total_rewards += reward
        self.steps += 1

        # if np.any([done, truncated]):
        #     done = True

        #print reward, done, truncated, self.steps in a fancy way
        # print(f"reward: {reward}, done: {done}, truncated: {truncated}, steps: {self.steps}")


        if self.max_steps > self.steps:
            truncated = False
        if done or truncated:
            info['returns/episodic_reward'] = self.total_rewards
            # if self.total_rewards != -1000:
            #     self.total_rewards =0

            info['returns/episodic_length'] = self.steps
            # self.total_rewards = 0
            # self.steps = 0
        else:
            info['returns/episodic_reward'] = None
            info['returns/episodic_length'] = None

        return obs, reward, done, truncated, info
    def reset(self, *, seed=73060, options=None):
        # super().__init__(env)
        self.total_rewards = 0
        self.steps = 0
        obs, _ = self.env.reset(seed=seed, options=options)
        return obs, _

def flatten_fullview_wrapperWrapper(env, reward_reg=5000, env_max_step=5000):
    env.max_steps = env_max_step
    env = FullyObsWrapper(env)
    env = FlattenWrapper(env)
    env = ReturnWrapper_wargs(env, reward_reg=reward_reg,  max_steps=env_max_step)
    return env
