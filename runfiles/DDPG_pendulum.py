import sys
sys.path.append('/Users/indurance/PycharmProjects/MasterProject')
import gym
import tensorflow as tf
import time
from envs.modified_pendulum import PendulumEnv
from agents.DDPG import DDPG
import numpy as np
from agents.model_learner import Model_Learner
import os
import utilits
from agents.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise


#####################  hyper parameters  ####################

MAX_EPISODES = 300
MAX_EP_STEPS = 200
MEMORY_CAPACITY = int(1e4)

#########################  training  ########################
# ENV_NAME = 'Pendulum-v0'
# ENV_NAME = 'Hopper-v2'
# ENV_NAME = 'Walker2d-v2'
# ENV_NAME = 'Swimmer-v2'
# ENV_NAME = 'InvertedPendulum-v2'
# ENV_NAME = 'Reacher-v2'
# env = gym.make(ENV_NAME)
env = PendulumEnv()
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

######### data directory setup ###########
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
exp_name = '{0}_{1}_{2}'.format('DDPG',
                                'Pendulum',
                                time.strftime("%d-%m-%Y_%H-%M-%S"))
exp_dir = os.path.join(data_dir, exp_name)
assert not os.path.exists(exp_dir), \
    'Experiment directory {0} already exists. Either delete the directory, or run the experiment with a different name'.format(
        exp_dir)
os.makedirs(exp_dir, exist_ok=True)
# print(exp_dir)
# wait = input("PRESS ENTER TO CONTINUE.")

########################################
######### the learning agent ###########
########################################
ddpg = DDPG(a_dim, s_dim, a_bound, MEMORY_CAPACITY=MEMORY_CAPACITY)

########################################
######### the learning model ###########
########################################
learned_model = Model_Learner(env,
                              num_actions=a_dim,
                              num_features=s_dim)

########## exploration noise ###########
#--- OrnsteinUhlenbeck Noise ----#
# var = 1  # control exploration
# noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(a_dim), sigma=float(var) * np.ones(a_dim))
#--- Normal Noise ----#
var = 1
noise = NormalActionNoise(mu=np.zeros(a_dim), sigma=float(var) * np.eye(a_dim))

t1 = time.time()
training_flag = True
iteration_counter = 0
reward_list = []
RENDER = False

N_plan = 1
for i in range(MAX_EPISODES):
    s = env.reset([np.pi, 0])
    s_eval = s
    ep_reward = 0
    ep_reward_eval = 0
    plan_flag = bool(i % N_plan != 0)  # if use the learned model to plan
    ep_type = ['Ture Environment', 'Learned Model'][int(plan_flag)]
    # noise.reset(sigma=var)
    noise.reset(sigma=0)

    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s) + noise()
        a = np.clip(a, -2, 2)  # add randomness to action selection for exploration

        if plan_flag == False:
            s_, r, done, info = env.step(a)
            learned_model.experience_dataset.add_data_pair(s, a, s_, r, done)
            # print('next state: {}'.format(s_.shape))
            # print('next state: {}'.format(r.shape))
            a_eval = ddpg.choose_action(s_eval)
            s_next_eval, r_eval = learned_model.simulated_step(s_eval, a_eval)
            ep_reward_eval += r_eval
            s_eval = s_next_eval
        else:
            s_, r = learned_model.simulated_step(s, a)
            s_ = s_.ravel()
            r = float(r)
            ep_reward_eval += r
            # print('next state: {}'.format(s_.shape))
            # print('next state: {}'.format(r.shape))

        # print('s:', s, 'a:', a, 'r:', r / 10, 's_:', s_)
        ddpg.store_transition(s, a, r / 10, s_)
        ddpg.learn()

        # if ddpg.pointer > MEMORY_CAPACITY:
        #     var *= .9995    # decay the action randomness

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS - 1:
            print('---------------------------------')
            print('######### Episode Info ##########')
            print('Episode:', i, 'Type:', ep_type, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:
            #   RENDER = True
            break

    # training_flag = bool(np.absolute((ep_reward - ep_reward_eval) / ep_reward) > 0.2) and ep_reward < -600
    training_flag = False

    if plan_flag == False:
        iteration_counter += 1
        # print('Simulated Reward: {}'.format(ep_reward_eval))
        # print('Model Uncertainty: {}'.format(np.absolute((ep_reward - ep_reward_eval) / ep_reward)))
        reward_list.append([iteration_counter, int(ep_reward)])
        np.savetxt(exp_dir + '/epsiode_data', reward_list, fmt='%4d')
        utilits.save_single_reward_curve1(np.array(reward_list)[:, 1], exp_dir)

    if ddpg.pointer > MEMORY_CAPACITY and var > 0.1:
        var *= .995    # decay the action randomness

    if training_flag:
        learned_model.model_training(training_epochs=10)
        learned_model.reward_model_training(training_epochs=5)
    ddpg.model_save(filepath=exp_dir)
    print('######### Episode End ###########')

print('Running time: ', time.time() - t1)