import sys
sys.path.append('/Users/indurance/PycharmProjects/MasterProject')
import gym
import tensorflow as tf
import time
from envs.modified_pendulum import PendulumEnv
from envs.modified_hopper import HopperEnv
from envs.modified_swimmer import SwimmerEnv
from envs.simulated_circular_pendulum import CircularPendulumEnv, CircularPendulumEnv2
from envs.modified_cheetah import HalfCheetahTrackingEnv
from envs.modified_inv_double_pendulum import InvertedDoublePendulumEnv
from envs.modified_rob_inv_pendulum import RoboschoolInvertedPendulum
from envs.modified_inv_pendulum import InvertedPendulumEnv
from envs.modified_hopper import HopperEnv
from agents.DDPG import DDPG
from agents.DDPG_v2 import DDPG_v2
import numpy as np
from agents.model_learner import Model_Learner
import os

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 500
MEMORY_CAPACITY = int(1e4)

#########################  training  ########################
RENDER = True
# ENV_NAME = 'Pendulum-v0'
# ENV_NAME = 'Hopper-v2'
# ENV_NAME = 'Walker2d-v2'
# ENV_NAME = 'Swimmer-v2'
# ENV_NAME = 'InvertedPendulum-v2'
# ENV_NAME = 'Reacher-v2'
# ENV_NAME = 'HalfCheetah-v2'
# env = gym.make(ENV_NAME)

# env = PendulumEnv()
# env = InvertedPendulumEnv()
# env = InvertedDoublePendulumEnv()
env = RoboschoolInvertedPendulum(swingup=True)
# env = SwimmerEnv()
# env = HopperEnv()
# env = HalfCheetahEnv()
# env = HalfCheetahTrackingEnv()
# env = CircularPendulumEnv2(episode_length=500, live_plot=True)
# env = CircularPendulumEnv(episode_length=500, live_plot=True)
# env = env.unwrapped
env.seed(1)

class argument_class(object):
    def __init__(self):
        self.env_name = 'Invert-v2'
        self.num_episodes = int(1e3)
        self.seed = 1
        self.num_steps = 500
        self.gamma = 0.99
        self.tau = 0.01
        self.critic_lr = 5e-4
        self.actor_lr = 1e-4
        self.batch_size = 64
        self.replayBuffer_size = int(2e4)

args = argument_class()

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

########################################
######### the learning agent ###########
########################################
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
ddpg = DDPG_v2(sess, env, args)

'''optimal performance'''
# path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data/DDPG_Pendulum_12-07-2019_17-57-51'
'''sub-optimal with oscillations'''
# path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data/MB_DDPG_Pendulum_12-07-2019_18-05-44'
'''optimal performance'''
# path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data/MB_DDPG_Pendulum_14-07-2019_17-40-06'
# path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data/DDPG_Swimmer_13-07-2019_17-47-46'
# path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data/DDPG2_HalfCheetah_20-07-2019_18-00-12'
'''great perforamce for circular pendulum'''
# path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data/DDPG2_CircularPendulum_31-07-2019_12-54-33'
'''great performance for half cheetah'''
# path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data_new/MB_DDPG2_HalfCheetahTrack_30-08-2019_14-45-28'
'''argument path'''
path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data_new/MB_DDPG2_RoboInvertedPendulum3_01-09-2019_17-00-07'
ddpg.model_load(filepath=path)
s = env.reset()
n_eps = 100
n_steps = 500

for eps in range(n_eps):

    # s = env.reset([0, np.pi - 0.1, 0, 0])
    s = env.reset()
    episode_reward = 0
    for steps in range(n_steps):
        a = ddpg.choose_action(s)
        # a = [0]
        a = np.clip(a, -2, 2)
        # a = 16.3555 * (s[1] - np.pi) + 2.0963 * s[3] * 10 - 1.7705 * s[2] * 10
        # a = - 2.179 * s[0] + 16.3555 * (s[1] - np.pi) - 1.7705 * s[2] * 10 + 2.0963 * s[3] * 10
        # print('action: {}'.format(a))
        # print('state: {}'.format(s))
        # a = 3
        s_, r, done, info = env.step(a)
        s = s_
        env.render()
        episode_reward += r/10
        # if done:
        #     break
    print('Episode {} ends with reward: {}'.format(eps, episode_reward))
    print('------------------------------')