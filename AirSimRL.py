import setup_path
import airsim

import numpy as np
import os
import tempfile
import pprint
import cv2

import gym 
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

#code adapted from: https://github.com/nicknochnack/ReinforcementLearningCourse/blob/main/Project%203%20-%20Custom%20Environment.ipynb

class FlightEnv(Env):
    """
    This is the environment that we use for reinforcement learning.
    The idea is that the drone will learn to stay within a certain altitude range
    (37 to 39 inclusive) where its reward will be maximized.
    """
    def __init__(self):
        """
        Constructor method
        """
        # Actions we can take (go down, stay at same altitude, or go up)
        self.action_space = Discrete(3)
        # altitude array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start altitude
        self.state = 38 + random.randint(-3,3)
        # Set flight length
        self.flight_length = 60
        
    def step(self, action):
        """
        One step of reinforcement learning. 

        :param action: the action that the drone should take (either 0, 1, or 2),
        0 means go down 1 unit, 1 means stay at the same altitude, and 2 means
        go up 1 unit
        :type action: int
        :return: step information
        :rtype: tuple
        """
        # Apply action
        # 0 -1 = -1 altitude
        # 1 -1 = 0 
        # 2 -1 = 1 altitude
        self.state += action -1 
        # Reduce flight length by 1
        self.flight_length -= 1 
        
        # Calculate reward
        if self.state >=37 and self.state <=39: 
            reward =1 
        else: 
            reward = -1 
        
        # Check if flight is done
        if self.flight_length <= 0: 
            done = True
        else:
            done = False
        
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        """
        Renders the environment (we don't use this since we use AirSim as the display)
        """
        # Implement viz
        pass
    
    def reset(self):
        """
        Resets the environment.

        :return: initial state (altitude)
        :rtype: numpy array of floats
        """
        # Reset altitude
        self.state = np.array([38 + random.randint(-3,3)]).astype(float)
        # Reset flight time
        self.flight_length = 60 
        return self.state



env=FlightEnv()


log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

#load saved model (see jupyter notebook for how we trained this model)
model = model.load('PPO')
# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

gps_data = client.getGpsData()
s = pprint.pformat(gps_data)
print("gps_data: %s" % s)

airsim.wait_key('Press any key to takeoff')
print("Taking off...")
client.armDisarm(True)
client.takeoffAsync().join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

#only do 1 episode
episodes = 1
for episode in range(1, episodes+1):
    #reset environment and get initial observation
    obs = env.reset()
    #move drone to this initial observation (altitude)
    client.moveToPositionAsync(0, 0, -(obs[0]), 5).join()
    done = False
    #keep track of total reward for the episode
    score = 0
    while not done:
        #predict action based on the observation (altitude)
        action, _ = model.predict(obs)
        #move to drone based on the returned action
        client.moveToPositionAsync(0, 0, -(env.state[0]) + ((-(action - 1)) * 5), 5).join()
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

env.close()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to reset to original state')

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)