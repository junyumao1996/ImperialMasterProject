import sys
sys.path.append('/Users/indurance/PycharmProjects/MasterProject')
import gym
import tensorflow as tf
import time
from envs.modified_pendulum import PendulumEnv, PendulumEnv2
from envs.modified_hopper import HopperEnv
from envs.modified_swimmer import SwimmerEnv
from envs.simulated_circular_pendulum import CircularPendulumEnv
from envs.modified_inv_double_pendulum import InvertedDoublePendulumEnv
from envs.modified_rob_inv_pendulum import RoboschoolInvertedPendulum
from agents.DDPG import DDPG
from agents.DDPG_v2 import DDPG_v2
import numpy as np
import matplotlib.pyplot as plt
from agents.model_learner import Model_Learner
import os

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
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
env = PendulumEnv()
env = PendulumEnv2()
# env = RoboschoolInvertedPendulum(swingup=True)
# env = InvertedDoublePendulumEnv()
# env = SwimmerEnv()
# env = HopperEnv()
# env = CircularPendulumEnv(episode_length=MAX_EP_STEPS)
env = env.unwrapped
env.seed(1)



s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

########################################
######### the learning agent ###########
########################################
ddpg = DDPG(a_dim, s_dim, a_bound, MEMORY_CAPACITY=MEMORY_CAPACITY)

'''optimal performance'''
# path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data/DDPG_Pendulum_12-07-2019_17-57-51'
'''sub-optimal with oscillations'''
# path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data/MB_DDPG_Pendulum_12-07-2019_18-05-44'
'''optimal performance'''
path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data/MB_DDPG_Pendulum_14-07-2019_17-40-06'
# path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data/DDPG_Swimmer_13-07-2019_17-47-46'
# path = '/Users/indurance/PycharmProjects/MasterProject/runfiles/data/DDPG_RoboInvertedPendulum_29-07-2019_22-37-14'
ddpg.model_load(filepath=path)
s = env.reset()
n_eps = 1
n_steps = 500
th_list = []
a_list = []
theta_dot_list = []

for eps in range(n_eps):
    # s = env.reset(pos=[0, np.pi-0.01, 0, 0])
    s, th = env.reset([np.pi, 0])
    # s, th = env.reset()
    # s = env.reset()
    episode_reward = 0
    for steps in range(n_steps):
        a = ddpg.choose_action(s)
        # a = -6
        # a = env.action_space.sample()
        s_, th_, r, done, info = env.step(a)
        s = s_
        th = th_
        env.render()
        episode_reward += r
        th_list.append(th)
        a_list.append(a)
        theta_dot_list.append(s[2])
        # if done:
        #     break
    print('Episode {} ends with reward: {}'.format(eps, episode_reward))
    print('------------------------------')

t = 0.05 * np.arange(n_steps)
fig = plt.figure()
ax_2 = fig.add_subplot(111)
_ = ax_2.plot(t, a_list, t, theta_dot_list, t, th_list)
_ = ax_2.set_xlabel("time (s)")
_ = ax_2.set_ylabel("Torque (Nm) / Angular Speed (rad/s)/ Angle (rad)")
_ = ax_2.set_title("Optimal Policy from RL")
_ = ax_2.legend(['control', 'theta_dot', 'theta'], loc='upper right')

np.save('th', np.array(th_list))
np.save('th_dot', np.array(theta_dot_list))

plt.grid()
plt.show()
# np.savetxt('/epsiode_data', reward_list, fmt='%4d')