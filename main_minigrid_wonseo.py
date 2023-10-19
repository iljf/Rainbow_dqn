# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle

import gymnasium as gym

# import wrappers for atari games
# from gym.wrappers import AtariPreprocessing
# from gym.wrappers import FrameStack
# from gym_minigrid.wrappers import *
from minigrid.wrappers import *

import numpy as np
import torch
from tqdm import trange

from agent import Agent
# from env import Env
from memory import ReplayMemory
from test import test, test_minigrid
import wandb

from util_wrapper import *



# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='MiniGrid-Empty-5x5-v0', help='minigrid')
# parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--T-max', type=int, default=int(5e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
# parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=1, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='mlp', choices=['canonical', 'data-efficient', 'mlp'], metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=128, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-2, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=2, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
# parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=1, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8e3), metavar='τ', help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=0, metavar='VALUE', help='Reward clipping (0 to disable)')
# parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--learning-rate', type=float, default=0.03, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
# parser.add_argument('--learn-start', type=int, default=int(20e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--learn-start', type=int, default=int(10e3), metavar='STEPS', help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
# parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
# parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
# parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--evaluation-size', type=int, default=1000, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
parser.add_argument('--env-max-step', type=int, default=5000)

# Setup
args = parser.parse_args()

# wandb intialize
wandb.init(project="Rainbow_dk_test",
           name=args.game + str(datetime.now()),
           config=args.__dict__
           )

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('./results', args.id)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = args.enable_cudnn
else:
    args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  else:
    with bz2.open(memory_path, 'rb') as zipped_pickle_file:
      return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
  if disable_bzip:
    with open(memory_path, 'wb') as pickle_file:
      pickle.dump(memory, pickle_file)
  else:
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
      pickle.dump(memory, zipped_pickle_file)

# if np.random.rand() < eps:
#     # do action that is random.

# Environment
env = gym.make(args.game, render_mode = 'rgb_array')
# env = gym.make(args.game, render_mode = 'human')
# env = RGBImgPartialObsWrapper(env) # Get pixel observations
env = flatten_fullview_wrapperWrapper(env, reward_reg=5000, env_max_step=args.env_max_step)


# env = AtariPreprocessing(env, screen_size=84)
# env = FrameStack(env, 2)


action_space = env.action_space.n

# Agent
dqn = Agent(args, env)

# If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
  if not args.memory:
    raise ValueError('Cannot resume training without memory save path. Aborting...')
  elif not os.path.exists(args.memory):
    raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

  mem = load_memory(args.memory, args.disable_bzip_memory)

else:
  mem = ReplayMemory(args, args.memory_capacity)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size)
T, done = 0, True
while T < args.evaluation_size:
    if done or truncated:
        state, _ = env.reset(seed=42)
        state = torch.Tensor(state)
        # state = state.unsqueeze(0)
        done = False

    next_state, r, done, truncated, info = env.step(np.random.randint(0, action_space))
    next_state = torch.Tensor(next_state)
    # next_state = next_state.unsqueeze(0)
    val_mem.append(state.unsqueeze(0), None, None, done)
    state = next_state
    T += 1

    wandb.log({'Replay_memory/reward':r})
    # wandb.log({'wandb 저장명': 값})

if args.evaluate:
  dqn.eval()  # Set DQN (online network) to evaluation mode
  avg_reward, avg_Q = test_minigrid(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
  print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))


else:

  # Training loop
  dqn.train()
  done = True
  for T in trange(1, args.T_max + 1):
    if done or truncated:
      # print episode rewards
      print('episode reward: ', env.total_rewards)
      state, _ = env.reset(seed=42)
      state = torch.Tensor(state).to(args.device)
      eps = 0.05
      # done, truncated = False, False
      # state = state.unsqueeze(0)

    if T % args.replay_frequency == 0:
      dqn.reset_noise()  # Draw a new set of noisy weights

    if np.random.rand() < eps:
        action = np.random.randint(0, action_space)
    else:
        action = dqn.act(state)  # Choose an action greedily (with noisy weights)
    # with torch.no_grad():
    #     value = dqn.online_net(state)(action)
    next_state, reward, done, truncated, info = env.step(action)
    next_state = torch.Tensor(next_state).to(args.device)
    # next_state = next_state.unsqueeze(0)
    # Step

    if args.reward_clip > 0:
      reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
    # mem.append(state, action, reward, done)  # Append transition to memory
    # if np.prod(state.shape)>75:
    #     print('w')
    mem.append(state.unsqueeze(0), action, reward, done)  # Append transition to memory

    wandb.log({'training/reward': reward
               })



    #check if mem is corrupted
    # for i in range(mem.transitions.data.shape[0]):
    #     if mem.transitions.data[i] is not None:
    #         # print(mem.transitions.data[i].state.shape)
    #         if np.prod(mem.transitions.data[i].state.shape) > 75:
    #             print('wtf')

    # ?(mem.transition.data[i], 'state')
    # mem.tan


    # Train and test
    if T >= args.learn_start:
      mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

      if T % args.replay_frequency == 0:
        dqn.learn(mem)  # Train with n-step distributional double-Q learning

      if T % args.evaluation_interval == 0:
        dqn.eval()  # Set DQN (online network) to evaluation mode
        avg_reward, avg_Q = test_minigrid(args, T, dqn, val_mem, metrics, results_dir)  # Test
        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
        dqn.train()  # Set DQN (online network) back to training mode

        wandb.log({'eval/reward' : reward,
                   'eval/Average_reward': avg_reward,
                   'eval/timestep': T,
                   'eval/Q-value': avg_Q
                   })

        # If memory path provided, save it
        if args.memory is not None:
          save_memory(mem, args.memory, args.disable_bzip_memory)

      # Update target network
      if T % args.target_update == 0:
        dqn.update_target_net()

      # Checkpoint the network
      if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
        dqn.save(results_dir, 'checkpoint.pth')

    state = next_state

env.close()
