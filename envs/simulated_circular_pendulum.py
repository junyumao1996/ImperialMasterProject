from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class CircularPendulumEnv(object):
    def __init__(self, g=10.0, dt=0.02, episode_length=200, live_plot=False):
        self.max_speed_theta = 50
        self.max_speed_alpha = 100
        self.max_voltage = 2
        self.dt = dt
        self.g = g
        self.viewer = None

        # high = np.array([np.pi, 2 * np.pi, self.max_speed_theta, self.max_speed_alpha]) # mind the angle conversion here
        # low = np.array([-np.pi, 0,  -self.max_speed_theta, -self.max_speed_alpha])      # mind the angle conversion here

        high = np.array([1, 1, 1, 1, self.max_speed_theta, self.max_speed_alpha])  # mind the angle conversion here
        low = np.array([-1, -1, -1, -1, -self.max_speed_theta, -self.max_speed_alpha])  # mind the angle conversion here
        self.action_space = spaces.Box(low=-self.max_voltage, high=self.max_voltage, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.theta_trajectory = []
        self.alpha_trajectory = []
        self.time_window = episode_length * dt

        self.seed()
        if live_plot:
          plt.ion()  ## Note this correction

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        theta, alpha, theta_dot, alpha_dot = self.state # th := theta
        u = np.clip(u, -self.max_voltage, self.max_voltage)
        voltage = u * 10

        dt = self.dt

        next_theta = theta + theta_dot * dt
        next_alpha = alpha + alpha_dot * dt

        eta1 = -0.1284 * voltage - 0.07143 * theta_dot + 0.004266 * np.sin(alpha) * np.square(
            alpha_dot) - 0.006149 * np.sin(alpha) * alpha_dot * theta_dot * np.cos(alpha)
        eta2 = 0.1938 * np.sin(alpha) + 0.0024 * alpha_dot - 0.003075 * np.sin(alpha) * np.square(theta_dot) * np.cos(
            alpha)
        eta3 = 0.00003135 * np.square(np.cos(alpha)) - 0.00004700
        theta_dot_dot = -(0.004275 * eta1 + 0.004266 * np.cos(alpha) * eta2) / eta3
        alpha_dot_dot = (0.004266 * np.cos(alpha) * eta1 - (
                0.003075 * np.square(np.cos(alpha)) - 0.01099) * eta2) / eta3

        next_theta_dot = theta_dot + theta_dot_dot * dt
        next_alpha_dot = alpha_dot + alpha_dot_dot * dt
        next_theta_dot = np.clip(next_theta_dot, -self.max_speed_theta, self.max_speed_theta)
        next_alpha_dot = np.clip(next_alpha_dot, -self.max_speed_alpha, self.max_speed_alpha)

        costs = 0
        costs = 0.2 * angle_normalize_theta(theta)**2
        costs += (angle_normalize_alpha(alpha)-np.pi)**2
        costs += 0 * (theta_dot**2 + alpha_dot**2)
        costs += .001*(u**2)
        costs = float(costs)
        ###### output the state ######
        self.state = np.array([next_theta, next_alpha, next_theta_dot, next_alpha_dot])

        ###### append the state trajectory ######
        self.theta_trajectory.append(theta)
        self.alpha_trajectory.append(alpha)

        return self.get_obs(self.state), -costs, False, {}

    def get_obs(self, state):
        return np.array([np.cos(state[0]), np.sin(state[0]), np.cos(state[1]), np.sin(state[1]),
                         state[2]/10, state[3]/10])

    def reset(self, pos=None, noise=False):
        # high = np.array([np.pi, 1])
        # self.state = self.np_random.uniform(low=-high, high=high)
        # self.state = np.array([np.pi, 1])
        if pos != None:
            self.state = np.array(pos)
        else:
            self.state = np.array([0, 0, 0, 0])

        if noise == True:
            self.state += np.random.uniform(low=-0.05, high=0.05, size=self.state.shape)
        else:
            pass

        self.last_u = None
        self.theta_trajectory = []
        self.alpha_trajectory = []

        plt.close()
        self.fig = plt.figure()
        self.ax_theta = self.fig.add_subplot(121)
        self.ax_alpha = self.fig.add_subplot(122)
        self.ax_theta.grid()
        self.ax_alpha.grid()
        return self.get_obs(self.state)

    def render(self):
        time = len(self.theta_trajectory) * self.dt
        time_length = np.arange(len(self.theta_trajectory)) * self.dt
        self.ax_theta.set_xlim([0, self.time_window])
        self.ax_alpha.set_xlim([0, self.time_window])
        # self.ax_theta.axis([0, 10, -1.5 * np.pi, 1.5 * np.pi])
        # self.ax_alpha.axis([0, 10, -2.5 * np.pi, 1.5 * np.pi])
        theta_tra = np.unwrap(self.theta_trajectory)
        alpha_tra = np.unwrap(self.alpha_trajectory)
        self.ax_theta.plot(time_length, theta_tra, 'b')
        self.ax_theta.set_xlabel("time (s)")
        self.ax_theta.set_ylabel("theta (rad)")

        self.ax_alpha.plot(time_length, alpha_tra, 'r')
        self.ax_alpha.set_xlabel("time (s)")
        self.ax_alpha.set_ylabel("alpha (rad)")
        plt.show()
        plt.pause(0.00001)

##############################################
######### changed version for MB task ########
##############################################

class CircularPendulumEnv2(object):
    def __init__(self, g=10.0, dt=0.02, episode_length=200, live_plot=False):
        self.max_speed_theta = 50
        self.max_speed_alpha = 100
        self.max_voltage = 2
        self.dt = dt
        self.g = g
        self.viewer = None

        # high = np.array([np.pi, 2 * np.pi, self.max_speed_theta, self.max_speed_alpha]) # mind the angle conversion here
        # low = np.array([-np.pi, 0,  -self.max_speed_theta, -self.max_speed_alpha])      # mind the angle conversion here

        high = np.array([1, 1, self.max_speed_theta, self.max_speed_alpha])  # mind the angle conversion here
        low = np.array([-1, -1, -self.max_speed_theta, -self.max_speed_alpha])  # mind the angle conversion here
        self.action_space = spaces.Box(low=-self.max_voltage, high=self.max_voltage, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.theta_trajectory = []
        self.alpha_trajectory = []
        self.time_window = episode_length * dt

        self.seed()
        if live_plot:
          plt.ion()  ## Note this correction

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        theta, alpha, theta_dot, alpha_dot = self.state # th := theta
        u = np.clip(u, -self.max_voltage, self.max_voltage)
        voltage = u * 10

        dt = self.dt

        next_theta = theta + theta_dot * dt
        next_alpha = alpha + alpha_dot * dt

        eta1 = -0.1284 * voltage - 0.07143 * theta_dot + 0.004266 * np.sin(alpha) * np.square(
            alpha_dot) - 0.006149 * np.sin(alpha) * alpha_dot * theta_dot * np.cos(alpha)
        eta2 = 0.1938 * np.sin(alpha) + 0.0024 * alpha_dot - 0.003075 * np.sin(alpha) * np.square(theta_dot) * np.cos(
            alpha)
        eta3 = 0.00003135 * np.square(np.cos(alpha)) - 0.00004700
        theta_dot_dot = -(0.004275 * eta1 + 0.004266 * np.cos(alpha) * eta2) / eta3
        alpha_dot_dot = (0.004266 * np.cos(alpha) * eta1 - (
                0.003075 * np.square(np.cos(alpha)) - 0.01099) * eta2) / eta3

        next_theta_dot = theta_dot + theta_dot_dot * dt
        next_alpha_dot = alpha_dot + alpha_dot_dot * dt
        next_theta_dot = np.clip(next_theta_dot, -self.max_speed_theta, self.max_speed_theta)
        next_alpha_dot = np.clip(next_alpha_dot, -self.max_speed_alpha, self.max_speed_alpha)

        costs = 0
        costs = 0.2 * angle_normalize_theta(theta)**2
        costs += (angle_normalize_alpha(alpha)-np.pi)**2
        costs += 0 * (theta_dot**2 + alpha_dot**2)
        costs += .001*(u**2)
        costs = float(costs)
        ###### output the state ######
        self.state = np.array([next_theta, next_alpha, next_theta_dot, next_alpha_dot])

        ###### append the state trajectory ######
        self.theta_trajectory.append(theta)
        self.alpha_trajectory.append(alpha)

        return self.state_normalize(self.state), -costs, False, {}

    @staticmethod
    def state_normalize(state):
        return np.array([state[0], state[1], state[2]/10, state[3]/10])

    def get_obs(self, state):
        return np.array([np.cos(state[0]), np.sin(state[0]), np.cos(state[1]), np.sin(state[1]),
                         state[2]/10, state[3]/10])

    def reset(self, pos=None, noise=False):
        # high = np.array([np.pi, 1])
        # self.state = self.np_random.uniform(low=-high, high=high)
        # self.state = np.array([np.pi, 1])
        if pos != None:
            self.state = np.array(pos)
        else:
            self.state = np.array([0, 0, 0, 0])

        if noise == True:
            self.state += np.random.uniform(low=-0.05, high=0.05, size=self.state.shape)
        else:
            pass

        self.last_u = None
        self.theta_trajectory = []
        self.alpha_trajectory = []

        plt.close()
        self.fig = plt.figure()
        self.ax_theta = self.fig.add_subplot(121)
        self.ax_alpha = self.fig.add_subplot(122)
        self.ax_theta.grid()
        self.ax_alpha.grid()
        return self.state_normalize(self.state)

    def render(self):
        time = len(self.theta_trajectory) * self.dt
        time_length = np.arange(len(self.theta_trajectory)) * self.dt
        self.ax_theta.set_xlim([0, self.time_window])
        self.ax_alpha.set_xlim([0, self.time_window])
        # self.ax_theta.axis([0, 10, -1.5 * np.pi, 1.5 * np.pi])
        # self.ax_alpha.axis([0, 10, -2.5 * np.pi, 1.5 * np.pi])
        theta_tra = np.unwrap(self.theta_trajectory)
        alpha_tra = np.unwrap(self.alpha_trajectory)
        self.ax_theta.plot(time_length, theta_tra, 'b')
        self.ax_theta.set_xlabel("time (s)")
        self.ax_theta.set_ylabel("theta (rad)")

        self.ax_alpha.plot(time_length, alpha_tra, 'r')
        self.ax_alpha.set_xlabel("time (s)")
        self.ax_alpha.set_ylabel("alpha (rad)")
        plt.show()
        plt.pause(0.00001)

def angle_normalize_theta(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def angle_normalize_alpha(x):
    return (x % (2*np.pi))

def state_normalize(state):
    s = state
    s[0] = angle_normalize_theta(s[0])
    s[1] = angle_normalize_alpha(s[1])
    return s