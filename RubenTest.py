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

import time


class FlightEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # altitude array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start altitude
        self.state = 38 + random.randint(-3,3)
        # Set flight length
        self.flight_length = 60
        
    def step(self, action):
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
        
        # Apply altitude noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset altitude
        self.state = np.array([38 + random.randint(-3,3)]).astype(float)
        # Reset flight time
        self.flight_length = 60 
        return self.state



env=FlightEnv()


log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

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

#print(f"TESTING THIS: {state.kinematics_estimated.position.z_val}")

episodes = 1
for episode in range(1, episodes+1):
    obs = env.reset()
    print(f"here it is: {obs[0]}")
    client.moveToPositionAsync(0, 0, -(obs[0]), 5).join()
    done = False
    score = 0
    while not done:
        #env.render()
        action, _ = model.predict(obs)
        client.moveToPositionAsync(0, 0, -(env.state[0]) + ((-(action - 1)) * 5), 5).join()
        
        obs, reward, done, info = env.step(action)
        print(done)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

env.close()
# airsim.wait_key('Press any key to move vehicle to (0, 0, 0) at 5 m/s')
# client.moveToPositionAsync(0, 0, 0, 5).join()

# client.hoverAsync().join()

# airsim.wait_key('Press any key to move vehicle to (10, -10, -10) at 5 m/s')
# client.moveToPositionAsync(10, -10, -10, 5).join()

# client.hoverAsync().join()

# airsim.wait_key('Press any key to move vehicle to (0, 0, 0) at 5 m/s')
# client.moveToPositionAsync(0, 0, 0, 5).join()

#client.hoverAsync().join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to reset to original state')

client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)

########################START HERE#######################

# import gym 
# from gym import Env
# from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
# import numpy as np
# import random
# import os
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.evaluation import evaluate_policy

# class FlightEnv(Env):
#     def __init__(self):
#         # Actions we can take, down, stay, up
#         self.action_space = Discrete(3)
#         # altitude array
#         self.observation_space = Box(low=np.array([0]), high=np.array([100]))
#         # Set start altitude
#         self.state = 38 + random.randint(-3,3)
#         # Set flight length
#         self.flight_length = 60
        
#     def step(self, action):
#         # Apply action
#         # 0 -1 = -1 altitude
#         # 1 -1 = 0 
#         # 2 -1 = 1 altitude
#         self.state += action -1 
#         # Reduce flight length by 1
#         self.flight_length -= 1 
        
#         # Calculate reward
#         if self.state >=37 and self.state <=39: 
#             reward =1 
#         else: 
#             reward = -1 
        
#         # Check if flight is done
#         if self.flight_length <= 0: 
#             done = True
#         else:
#             done = False
        
#         # Apply altitude noise
#         #self.state += random.randint(-1,1)
#         # Set placeholder for info
#         info = {}
        
#         # Return step information
#         return self.state, reward, done, info

#     def render(self):
#         # Implement viz
#         pass
    
#     def reset(self):
#         # Reset altitude
#         self.state = np.array([38 + random.randint(-3,3)]).astype(float)
#         # Reset flight time
#         self.flight_length = 60 
#         return self.state



# env=FlightEnv()

# env.reset()

# episodes = 5
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0 
    
#     while not done:
#         env.render()
#         action, _ = model.predict(obs)

#         #use airsim command to do the action
#         n_state, reward, done, info = env.step(action)

#         client.moveToPositionAsync(0, 0, , 5).join()
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()
