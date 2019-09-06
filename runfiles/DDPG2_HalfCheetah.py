import sys
sys.path.append('/Users/indurance/PycharmProjects/MasterProject')
import gym
import tensorflow as tf
import time
from envs.modified_pendulum import PendulumEnv
from agents.DDPG_v2 import DDPG_v2
import numpy as np
from agents.model_learner import Model_Learner
import os
import utilits
from agents.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

#####################  hyper parameters  ####################
class argument_class(object):
    def __init__(self):
        self.env_name = 'HalfCheetah-v2'
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
env = gym.make(args.env_name)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high


##################### data directory setup ###############
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
exp_name = '{0}_{1}_{2}'.format('DDPG2',
                                'HalfCheetah',
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

sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
ddpg = DDPG_v2(sess, env, args)

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
var = 3
noise = NormalActionNoise(mu=np.zeros(a_dim), sigma=float(var) * np.eye(a_dim))

t1 = time.time()
training_flag = True
iteration_counter = 0
reward_list = []
RENDER = False

N_plan = 1
for i in range(args.num_episodes):
    start = time.time()
    s = env.reset()
    s_eval = s
    ep_reward = 0
    ep_reward_eval = 0
    plan_flag = bool(i % N_plan != 0)  # if use the learned model to plan
    ep_type = ['Ture Environment', 'Learned Model'][int(plan_flag)]
    noise.reset(sigma=var)
    if var >= 0.1:
        var *= .995  # decay the action randomness

    for j in range(args.num_steps):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s) + noise()
        a = np.clip(a, -2, 2)  # add randomness to action selection for exploration

        if plan_flag == False:
            s_, r, done, info = env.step(a)
        else:
            s_, r = learned_model.simulated_step(s, a)
            s_ = s_.ravel()
            r = float(r)
            ep_reward_eval += r
            # print('next state: {}'.format(s_.shape))
            # print('next state: {}'.format(r.shape))

        ep_reward += r

        if j == args.num_steps - 1:
            end = time.time()
            print('---------------------------------')
            print('######### Episode Info ##########')
            print('Episode:', i, 'Time:', int(end-start), ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )

            ddpg.perceive(s, a, r * 0.1, s_, episode=i, ep_reward=ep_reward)
            if ep_reward > 500:
                # RENDER = True
                ddpg.model_save(filepath=exp_dir)
        else:
            ddpg.perceive(s, a, r * 0.1, s_)

        s = s_

    # training_flag = bool(np.absolute((ep_reward - ep_reward_eval) / ep_reward) > 0.2) and ep_reward < -600
    training_flag = False

    if plan_flag == False:
        iteration_counter += 1
        # print('Simulated Reward: {}'.format(ep_reward_eval))
        # print('Model Uncertainty: {}'.format(np.absolute((ep_reward - ep_reward_eval) / ep_reward)))
        reward_list.append([iteration_counter, int(ep_reward)])
        np.savetxt(exp_dir + '/epsiode_data', reward_list, fmt='%4d')
        utilits.save_single_reward_curve1(np.array(reward_list)[:, 1], exp_dir)

    if training_flag:
        learned_model.model_training(training_epochs=10)
        learned_model.reward_model_training(training_epochs=5)

    print('######### Episode End ###########')

print('Running time: ', time.time() - t1)

