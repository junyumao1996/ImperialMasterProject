import sys
sys.path.append('/Users/indurance/PycharmProjects/MasterProject')
import gym
import tensorflow as tf
import time
from envs.simulated_circular_pendulum import CircularPendulumEnv, CircularPendulumEnv2, state_normalize
from agents.DDPG_v2 import DDPG_v2
import numpy as np
from agents.model_learner import Model_Learner
import os
import utilits
from agents.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise

#####################  hyper parameters  ####################
class argument_class(object):
    def __init__(self):
        self.env_name = 'CircularPendulum2'
        self.num_episodes = int(300)
        self.seed = 1
        self.num_steps = 500
        self.imaginary_num_steps = 100
        self.gamma = 0.99
        self.tau = 0.01
        self.critic_lr = 5e-4
        self.actor_lr = 1e-4
        self.batch_size = 64
        self.replayBuffer_size = int(2e4)
        self.validation_length = 100

args = argument_class()
env = CircularPendulumEnv()

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
a_max = env.action_space.high
a_min = env.action_space.low
print('State space: Dimension {} Bounds {} - {}'.format(env.observation_space.shape, env.observation_space.low,
                                                       env.observation_space.high))
print('Action space: {0} - {1}'.format(a_min, a_max))

# input('pause')

##################### data directory setup ###############
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
exp_name = '{0}_{1}_{2}'.format('MB_DDPG2',
                                'CircularPendulum',
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
original_feature_dim = 6
learned_model = Model_Learner(env,
                              num_actions=a_dim,
                              num_features=original_feature_dim,
                              num_hidden_layers_d=3,
                              num_hidden_layers_r=2,
                              num_hidden_units_d=150)
# print('Pre-Train Models ...')
# # learned_model.rollouts_collection(learned_model.env, learned_model.experience_dataset, len_rollouts=200)
learned_model.model_training(training_epochs=60)
learned_model.reward_model_training(training_epochs=60)
learned_model.entire_model_testing()
learned_model.entire_model_testing_testdata()
print('Pre-Training Completed!')

input('pause')
########################################
########## exploration noise ###########
########################################
#--- OrnsteinUhlenbeck Noise ----#
# var = 1  # control exploration
# noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(a_dim), sigma=float(var) * np.ones(a_dim))
#--- Normal Noise ----#
var = 2
noise = NormalActionNoise(mu=np.zeros(a_dim), sigma=float(var) * np.eye(a_dim))


############ reward function ############
def reward_function(state, a):
    reward = np.cos(state[2])
    reward -= 0.2 * state[0] ** 2
    reward -= 0 * state[1] ** 2
    reward -= 0 * state[3] ** 2
    reward -= 0.001 * float(a) ** 2
    return reward

def episode_type_generator(ep_type_num, num_plan):
    ep_type_num += 1
    if ep_type_num >= (1 + num_plan):
        ep_type_num = 0
    return ep_type_num

def model_planning_number_generator(model_inaccuracy):
    num_plan = 0
    if model_inaccuracy < 0.15:
        num_plan = 2
    if model_inaccuracy < 0.1:
        num_plan += 2
    if model_inaccuracy < 0.05:
        num_plan += 2
    return num_plan


t1 = time.time()
training_flag = False
iteration_counter = 0
episode_counter = 0
reward_list = []
RENDER = False
ep_type_number = 0
model_uncertainty = 1.

N_plan = 0
while iteration_counter < args.num_episodes:
    start = time.time()
    ep_reward = 0
    segment_reward_real = 0
    segment_reward_eval = 0
    ep_type = ['Ture Environment', 'Learned Model'][int(ep_type_number != 0)]
    noise.reset(sigma=var)
    if var >= 0.05 and (ep_type_number == 0 or ep_type_number == 1):
        var *= .995  # decay the action randomness

    if ep_type_number == 0:
        s = env.reset()
        frozen_start_index = np.random.choice(args.num_steps - args.validation_length, size=1)
        frozen_start_indices = np.arange(args.validation_length) + int(frozen_start_index)

        for j in range(args.num_steps):
            if RENDER:
                env.render()
            if j in frozen_start_indices:
                if j == frozen_start_index:
                    s_eval = s
                a = ddpg.choose_action(s)            # no exploration noise
                a = np.clip(a, -2, 2)
                s_, r, done, info = env.step(a)

                learned_model.experience_dataset.add_data_pair(s, a, s_, r, done)
                ep_reward += r
                segment_reward_real += r

                a_eval = ddpg.choose_action(s_eval)
                s_next_eval, r_eval = learned_model.simulated_step(s_eval, a_eval)
                # r_eval = reward_function(s_eval, a_eval)               # direct use exposed reward function
                segment_reward_eval += r_eval
                s_eval = s_next_eval

            else:
                a = ddpg.choose_action(s) + noise()   # add randomness to action selection for exploration
                a = np.clip(a, -2, 2)
                s_, r, done, info = env.step(a)
                learned_model.experience_dataset.add_data_pair(s, a, s_, r, done)

                ep_reward += r


            if j == args.num_steps - 1:
                end = time.time()
                print('---------------------------------')
                print('######### Episode Info ##########')
                print('Episode:', episode_counter, '({})'.format(iteration_counter), 'Type:', ep_type, 'Time:', int(end - start),
                      ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var)
                print('Froze start index: {} '.format(int(frozen_start_index)) + 'Length: {}'.format(
                    len(frozen_start_indices)))

                ddpg.perceive(s, a, r * 0.1, s_, episode=episode_counter, ep_reward=ep_reward)
                if ep_reward > 500:
                    # RENDER = True
                    pass
            else:
                if j in frozen_start_indices:
                    ddpg.memory.store_transition(s, a, r * 0.1, s_)
                else:
                    ddpg.perceive(s, a, r * 0.1, s_)

            s = s_

        iteration_counter += 1
        model_uncertainty = np.absolute((segment_reward_real - segment_reward_eval) / segment_reward_real)
        N_plan = model_planning_number_generator(model_uncertainty)   # compute the number of planned rollouts
        training_flag = bool(model_uncertainty > 0.15) or model_uncertainty == None
        print('Real segment Reward: {}'.format(int(segment_reward_real)))
        print('Simulated segment Reward: {}'.format(int(segment_reward_eval)))
        print('Model Uncertainty: {}'.format(model_uncertainty))
        reward_list.append([iteration_counter, int(ep_reward)])
        np.savetxt(exp_dir + '/epsiode_data', reward_list, fmt='%4d')
        utilits.save_single_reward_curve1(np.array(reward_list)[:, 1], exp_dir)

        if training_flag:
            learned_model.model_training(training_epochs=10)
            learned_model.reward_model_training(training_epochs=10)
            training_flag = False

    else:
        s = learned_model.experience_dataset.random_state_generate()
        for j in range(args.imaginary_num_steps):
            # Add exploration noise
            a = ddpg.choose_action(s) + noise()
            a = np.clip(a, -1, 1)  # add randomness to action selection for exploration
            s_, r = learned_model.simulated_step(s, a)
            # r = reward_function(s, a)            # direct use exposed reward function
            s_ = s_.ravel()
            r = float(r)

            ep_reward += r

            if j == args.imaginary_num_steps - 1:
                end = time.time()
                print('---------------------------------')
                print('######### Episode Info ##########')
                print('Episode:', episode_counter, 'Type:', ep_type, 'Time:', int(end - start),
                      ' Reward:', int(ep_reward), 'Explore: %.2f' % var)

                ddpg.perceive(s, a, r * 0.1, s_, episode=episode_counter,
                              ep_reward=ep_reward)
                if ep_reward > 500:
                    # RENDER = True
                    pass
            else:
                ddpg.perceive(s, a, r * 0.1, s_)

            s = s_

        print('Type_number:', ep_type_number, 'Num_planning:', N_plan)

    ep_type_number = episode_type_generator(ep_type_number, N_plan)

    ddpg.model_save(filepath=exp_dir)
    print('######### Episode End ###########')
    episode_counter += 1