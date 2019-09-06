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
from envs.modified_rob_inv_pendulum import RoboschoolInvertedPendulum, RoboschoolInvertedPendulum_plot
from envs.modified_inv_pendulum import InvertedPendulumEnv
from envs.modified_hopper import HopperEnv
from agents.DDPG import DDPG
from agents.DDPG_v2 import DDPG_v2
import numpy as np
from agents.model_learner import Model_Learner
import os
import matplotlib.pyplot as plt

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
env = RoboschoolInvertedPendulum_plot(swingup=True)
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
n_eps = 1
n_steps = 500
a_list = []

for eps in range(n_eps):
    # s = env.reset(pos=[0, np.pi-0.01, 0, 0])
    s = env.reset()
    # s, th = env.reset()
    # s = env.reset()
    episode_reward = 0
    for steps in range(n_steps):
        a = ddpg.choose_action(s)
        # a = -6
        # a = env.action_space.sample()
        s_, s_vector, r, done, info = env.step(a)
        s = s_
        env.render()
        episode_reward += r
        a_list.append(a)
        if steps == 0:
            traj = s_vector
        else:
            traj = np.vstack((traj, s_vector))
        # if done:
        #     break
    print('Episode {} ends with reward: {}'.format(eps, episode_reward))
    print('------------------------------')


t = 0.05 * np.arange(n_steps)
fig = plt.figure()
ax_1 = fig.add_subplot(221)
_ = ax_1.plot(t, traj[:, 0], t, traj[:, 1])
_ = ax_1.set_xlabel("time (s)")
_ = ax_1.set_ylabel("Displacement (m) / Velocity (m/s)")
_ = ax_1.legend(['x', 'x_dot'], loc='upper right')
plt.grid()
plt.xlim((0, 25))

ax_2 = fig.add_subplot(223)
_ = ax_2.plot(t, traj[:, 2], t, traj[:, 3])
_ = ax_2.set_xlabel("time (s)")
_ = ax_2.set_ylabel("Angle (rad) / Angular Speed (rad/s)")
_ = ax_2.legend(['theta', 'theta_dot'], loc='upper right')
plt.grid()
plt.xlim((0, 25))

ax_3 = fig.add_subplot(2, 2, (2, 4))
_ = ax_3.plot(t, a_list)
_ = ax_3.set_xlabel("time (s)")
_ = ax_3.set_ylabel("Force (N)")
_ = ax_3.legend(['control'], loc='upper right')

plt.xlim((0, 25))

plt.grid()
plt.show()
# np.savetxt('/epsiode_data', reward_list, fmt='%4d')