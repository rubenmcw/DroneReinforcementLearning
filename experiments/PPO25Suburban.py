
#certain parts of the code below were initially borrowed from (but we have changed it A LOT): https://github.com/jemaw/gym-safety/blob/master/gym_safety/envs/gridnavigation.py

# -*- coding: utf-8 -*-
"""2DGridDroneMoreGoals.ipynb
"""


#you need to pip install stable_baselines3 so make sure you run this cell first

"""# 1. Import Dependencies"""

import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import random
import os
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import cv2
import matplotlib.pyplot as plt
from gym.utils import seeding

import math

from contextlib import contextmanager
import sys

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

DEFAULT_SEED = 42

NUM_ITERATIONS = 10

TIMESTEPS_FOR_TRAINING = 1000000


totalSuccessful = 0
totalnumSteps = 0

percentSuccessful = 0

fileName = "PPO25Suburban.txt"

f = open(fileName, "w")

EMPTY_CHAR = ' '
OBSTACLE_CHAR = '#'
PLAYER_CHAR = 'P'
GOAL_CHAR = 'G'

def makeTree(x, y, art):
  art[x][y] = OBSTACLE_CHAR
  art[x-1][y] = OBSTACLE_CHAR
  art[x+1][y] = OBSTACLE_CHAR
  art[x][y+1] = OBSTACLE_CHAR
  art[x][y-1] = OBSTACLE_CHAR


def build_asci_art(gridsize, rho):
        ########

        art = [[OBSTACLE_CHAR for i in range(50)] for j in range(50)]
        
        for r in range(0, 36):
          for c in range(23,29):
            art[r][c] = EMPTY_CHAR

        for r in range(8, 17):
          for c in range(11,37):
            art[r][c] = EMPTY_CHAR

        for r in range(3, 21):
          for c in range(11,15):
            art[r][c] = EMPTY_CHAR

        for r in range(2, 22):
          for c in range(33,38):
            art[r][c] = EMPTY_CHAR

        for r in range(2, 7):
          for c in range(33,49):
            art[r][c] = EMPTY_CHAR

        for r in range(7, 22):
          for c in range(45,49):
            art[r][c] = EMPTY_CHAR

        for r in range(16, 22):
          for c in range(38,45):
            art[r][c] = EMPTY_CHAR

        for r in range(3, 7):
          for c in range(0,15):
            art[r][c] = EMPTY_CHAR

        for r in range(7, 21):
          for c in range(0,4):
            art[r][c] = EMPTY_CHAR

        for r in range(16, 21):
          for c in range(4,11):
            art[r][c] = EMPTY_CHAR

        for r in range(30, 36):
          for c in range(9,23):
            art[r][c] = EMPTY_CHAR

        for r in range(36, 49):
          for c in range(8,13):
            art[r][c] = EMPTY_CHAR

        for r in range(36, 46):
          for c in range(16,22):
            art[r][c] = EMPTY_CHAR

        for r in range(39, 46):
          for c in range(22,26):
            art[r][c] = EMPTY_CHAR

        for r in range(37, 41):
          for c in range(26,48):
            art[r][c] = EMPTY_CHAR

        for r in range(41, 48):
          for c in range(26,32):
            art[r][c] = EMPTY_CHAR

        for r in range(43, 48):
          for c in range(32,48):
            art[r][c] = EMPTY_CHAR

        for r in range(41, 43):
          for c in range(44,48):
            art[r][c] = EMPTY_CHAR


        #green

        art[0][24] = GOAL_CHAR

        #red
        art[46][10] = PLAYER_CHAR

        art1 = []

        for r in art:
            art1.append(''.join(r))

        split_art = []
        for r in art1:
            for ch in r:
                split_art.append(ch)
        return np.array(split_art)

GLOBAL_ART = build_asci_art(50, 0.3).reshape((50, 50))

# Create a new random number generator with a different seed
rng = np.random.RandomState(4242)

for _ in range(NUM_ITERATIONS):

    generatedStartState = np.array([0, 0])

    generatedStartState[0] = rng.randint(0, 50)
    generatedStartState[1] = rng.randint(0, 50)

    surrounded = ((generatedStartState[0]-1) <0 or GLOBAL_ART[generatedStartState[0]-1][generatedStartState[1]] == OBSTACLE_CHAR) and ((generatedStartState[0]+1)>49 or GLOBAL_ART[generatedStartState[0]+1][generatedStartState[1]] == OBSTACLE_CHAR) and ((generatedStartState[1]-1)<0 or GLOBAL_ART[generatedStartState[0]][generatedStartState[1]-1] == OBSTACLE_CHAR) and ((generatedStartState[1]+1)>49 or GLOBAL_ART[generatedStartState[0]][generatedStartState[1]+1] == OBSTACLE_CHAR)


    while GLOBAL_ART[generatedStartState[0]][generatedStartState[1]] == OBSTACLE_CHAR or surrounded:
        print("IN HERE!")
        generatedStartState[0] = rng.randint(0, 50)
        generatedStartState[1] = rng.randint(0, 50)
        surrounded = ((generatedStartState[0]-1) <0 or GLOBAL_ART[generatedStartState[0]-1][generatedStartState[1]] == OBSTACLE_CHAR) and ((generatedStartState[0]+1)>49 or GLOBAL_ART[generatedStartState[0]+1][generatedStartState[1]] == OBSTACLE_CHAR) and ((generatedStartState[1]-1)<0 or GLOBAL_ART[generatedStartState[0]][generatedStartState[1]-1] == OBSTACLE_CHAR) and ((generatedStartState[1]+1)>49 or GLOBAL_ART[generatedStartState[0]][generatedStartState[1]+1] == OBSTACLE_CHAR)

    #environment

    class GridNavigationEnv(Env):
        """"
        Description:
            This game is used as simple environment for testing reinforcement learning algorithms that are built for safety.
            It is an unnoficial implementation of the one described in "A Lyapunov-based Approach to Safe Reinforcement Learning":
                https://arxiv.org/abs/1805.07708
            The goal for the player is to reach the end-state without violating constraints, here these constraints are modelled as
            immediate cost when hitting an obstacle and the constraint is to have the cumulative constraint cost below a certain threshold.
            The environment is stochastic, sometimes the agent performs a random action, instead of the chosen one.
        Parameters for customize game:
            seed (float):               seed for creating the game, can also be None for random seed
            rho (float):                density, meaning probability for obstacle, default: 0.3
            stochasticity (float):      probability for random action, default: 0.1
            img_observation (bool):     whether to use an img_observation or a one-hot encoding of the player position, default: false
            Note: v0 has gridsize 25 and v1 has grisize 60 for further customization use the customize_game function
        Observation:
            this environment has two modes: the observation is either a one-hot encoding of the player position or an RGB image of the complete board
        Actions:
            Type: Discrete(4)
            Num Action
            0 Go Down
            1 Go Left
            2       Go up
            3       Go right
        Reward:
            Every action has a negative fuel reward of -1. If the final state is reached the reward is 1000.
        Starting State:
            The agent always starts at the bottom right
        Episode Termination:
            - the agent reaches the goal state
        """
        def __init__(self, gridsize=50):
            #gridsize is 32 by default. We have a square grid so dimensions will be 32 by 32
            self.gridsize = gridsize
            #determines how many obstacles there are in the grid
            self.rho = 0.3
            #determines the likelihood that a random action will be performed instead of the one suggested by the model
            self.stochasticity = 0.1
            #coordinates of where the drone is
            self.state = None
            #starting coordinates of the drone (bottom right of grid)
            self.start_state = None
            #coordinates of where the goals are
            self.goal_states = None
            #whether to use an img_observation or a one-hot encoding of the player position, default: false (read above)
            self.img_observation = None

            self.maxNumSteps = None

            self.action_space = None
            self.observation_space = None

            #characters we will use when creating the grid to represent different types of cells
            self.EMPTY_CHAR = ' '
            self.OBSTACLE_CHAR = '#'
            self.PLAYER_CHAR = 'P'
            self.GOAL_CHAR = 'G'


            #reward matrix (gridsize by gridsize) which contains the reward for every empty cell on the grid (partial rewards). We calculate these rewards in the make_game method
            self.reward_matrix = None
            #display size for when we create a visual display for the user
            self.display_size = 256

            #colors for the different types of cells in the grid
            self.FG_COLORS = {self.EMPTY_CHAR: (0, 0, 0),  # normal background
                            self.GOAL_CHAR: (0, 1, 0),  # goal
                            self.PLAYER_CHAR: (1, 0, 0),   # player
                            self.OBSTACLE_CHAR: (0, 0, 1),      # obstacle
                        }


            #we use these numpy arrays to calculate the new state (coordinates of the drone) in the step method
            self.ACTIONS = {
                0: np.array([-1,0]),
                1: np.array([0,-1]),
                2: np.array([1,0]),
                3: np.array([0,1])
            }
            
            #initialize random number generator with the defualt seed and save it to self.np_random
            self.seed(DEFAULT_SEED)

            self.make_game()

        def customize_game(self, seed=DEFAULT_SEED, rho=None, stochasticity=None, img_observation=None):
            #initialize random number generator with seed and save it to self.np_random
            self.seed(seed)
            if not rho is None:
                self.rho = rho
            if not stochasticity is None:
                self.stochasticity = stochasticity
            if not img_observation is None:
                self.img_observation = img_observation
            self.make_game()

        #calculates euclidean distance of two lists
        def euclidean_distance(self, list1, list2):
          sum = 0
          for i in range(len(list1)):
            sum += (list1[i] - list2[i]) ** 2
          return math.sqrt(sum)

        #this will be replaced by Tolu's equation once he finishes it, the code below is just meant to be used temporarily
        def calculate_reward(self, dist_to_goal, dist_to_obstacle):
          reward = 0
          if dist_to_goal < 3:
            #if we are really close to a goal state then increase the reward by a lot
            reward += 3
          elif dist_to_goal < 6:
            #if we are really close to a goal state then increase the reward by a lot
            reward += 2.75
          elif dist_to_goal < 9:
            #if we are really close to a goal state then increase the reward by a lot
            reward += 2.25
          elif dist_to_goal < 12:
            #if we are kind of close to a goal state then increase the reward by some
            reward += 2
          elif dist_to_goal < 15:
            #if we are closeish to a goal state then increase the reward by a little
            reward += 1.75
          elif dist_to_goal < 18:
            #if we are closeish to a goal state then increase the reward by a little
            reward += 1.25
          elif dist_to_goal < 21:
            #if we are closeish to a goal state then increase the reward by a little
            reward += 1



          
          if dist_to_obstacle < 3:
            #if we are really close to an obstacle state, decrease the reward by a lot
            reward -= 3
          elif dist_to_obstacle < 6:
            #if we are kind of close to an obstacle state then decrease the reward by some
            reward -= 2.75
          elif dist_to_obstacle < 9:
            #if we are kind of close to an obstacle state then decrease the reward by some
            reward -= 2.25
          elif dist_to_obstacle < 12:
            #if we are kind of close to an obstacle state then decrease the reward by some
            reward -= 2
          elif dist_to_obstacle < 15:
            #if we are closeish to an obstacle state then decrease the reward by a little
            reward -= 1.75
          elif dist_to_obstacle < 18:
            #if we are closeish to an obstacle state then decrease the reward by a little
            reward -= 1.25
          elif dist_to_obstacle < 21:
            #if we are closeish to an obstacle state then decrease the reward by a little
            reward -= 1

          return reward




        def make_game(self):
            """Builds and returns a navigation game."""
            self.art = GLOBAL_ART

            #starting coordinates of the drone
            self.start_state = generatedStartState
            #coordinates of all of the goal states on the grid
            self.goal_states = np.argwhere(self.art == self.GOAL_CHAR).tolist()
            #the drone starts off at the start state
            self.state = self.start_state
            #coordinates of all of the obstacle states on the grid
            self.obstacle_states = np.argwhere(self.art == self.OBSTACLE_CHAR).tolist()

            self.maxNumSteps = 500

            """This is where we implement partial rewards. We give a reward for each empty cell on the grid.
            The reward we assign to each cell depends on its distance to the nearest goal state and its distance
            to the nearest obstacle state."""

            #initialize reward matrix
            self.reward_matrix = [[0 for i in range(self.gridsize)] for j in range(self.gridsize)]

            #implementing travis pseudocode

            numObstacles = len(self.obstacle_states)

            numGoals = len(self.goal_states)

            multiplyGoalsBy = numObstacles / float(numGoals)


            max_euclidian_distance = self.euclidean_distance([49,49],[0,0])

            for x in range(len(self.reward_matrix)):
              for y in range(len(self.reward_matrix)):
                if ([x, y] in self.obstacle_states) or ([x, y] in self.goal_states):
                  continue

                #PART 1: calculate total reward for how close we are to all goal states
                for goal_cell in self.goal_states:
                  #will scale distance between 1 and 0, where 1 is the best and 0 is the worst
                  distance_to_reward = self.euclidean_distance([x,y],goal_cell)

                  #reward is always going to be between 1 and 0, 1 is the best
                  reward = (max_euclidian_distance - distance_to_reward) / max_euclidian_distance

                  self.reward_matrix[x][y] += (math.e ** reward) * multiplyGoalsBy

                  #NOTE: you can tweak the reward so that it decreases faster or slower the farther you get from the goal cell
                  #another option
                  #self.reward_matrix[x][y] += math.e ** reward

                #PART 2: reduce reward for being close to obstacles
                for obstacle_cell in self.obstacle_states:
                  distance_to_obstacle = self.euclidean_distance([x,y],obstacle_cell)

                  #reward is always going to be between 1 and 0, 1 is the best
                  reward = (max_euclidian_distance - distance_to_obstacle) / max_euclidian_distance

                  self.reward_matrix[x][y] -= math.e ** reward  #-0.5 is the hyperparameter #hyperparameter < 0


            


            #create image of grid without the player in it
            self.art_img = self.get_art_img()


            self.action_space = Discrete(4)
            if self.img_observation:
                self.observation_space = Box(0, 1, shape=(self.gridsize, self.gridsize, 3), dtype=np.float32)
            else:
                self.observation_space = Box(0, 1, shape=(self.gridsize*self.gridsize,), dtype=np.float32)

        


        
        def sig(self, x):
            return 1/(1 + np.exp(-0.02*x))

        def get_art_img(self):
            """returns image of grid without the player (drone) in it"""
            img = np.zeros((self.gridsize, self.gridsize, 3))
            for i in range(self.gridsize):
                for j in range(self.gridsize):
                    if self.art[i,j] == self.PLAYER_CHAR or self.art[i,j] == self.EMPTY_CHAR:
                        amount = self.sig(self.reward_matrix[i][j])
                        img[i,j,:] = (amount, amount, amount)
                    else:
                        img[i,j,:] = self.FG_COLORS[self.art[i,j]]
            return img

        def reset(self):
            self.state = self.start_state
            if self.img_observation:
                obs = self.state_to_img()
            else:
                obs = self.state_to_oh()
            self.maxNumSteps = 500
            return obs

        def state_to_img(self):
            #insert the drone into the image and return the image
            img = self.art_img.copy()
            i, j = self.state
            img[i,j,:] = self.FG_COLORS[self.PLAYER_CHAR]
            return img

        def state_to_oh(self):
            #create a 1 hot encoded vector of the player position
            obs = np.zeros((self.gridsize*self.gridsize))
            state_i, state_j = self.state
            obs[state_i*self.gridsize + state_j] = 1
            return obs

        def step(self, action):
            #there is a percent chance that a random action will be taken instead of the one recommended by the model
            if np.random.binomial(1, self.stochasticity):
                action = np.random.randint(0,4)
            #used to prevent the overriding of reward after we have set it
            alreadySetReward = False
            #calculate the coordinates of where the drone is moving
            new_state = self.state+self.ACTIONS[action]

            #decompose in order to get x and y
            new_state_x, new_state_y = new_state

            #check to make sure where we are moving (new_state) is within the grid
            if not (new_state_x < 0 or \
              new_state_x >= self.gridsize or \
              new_state_y < 0 or \
              new_state_y >= self.gridsize):
              #check to make sure where we are moving isn't an obstacle state (not allowed to move into an obstacle state since there is an obstacle there)
                if new_state.tolist() in self.obstacle_states:
                  reward = -3
                  alreadySetReward = True
                else:
                    #valid location so move the drone there
                  self.state = new_state
            else:
              reward = -100
              alreadySetReward = True


            done = False
            info = {}

            self.maxNumSteps -= 1

            if alreadySetReward == False:
              if self.state.tolist() in self.goal_states:
                  done = True
                  info["successful"] = True
                  reward = 100000
              else:
                  #if the drone is at an empty cell, get that cell's partial reward from the reward matrix and set that to reward
                reward = self.reward_matrix[self.state[0]][self.state[1]] + (random.randint(-25, 25)/100)

            if done == False and self.maxNumSteps == 0:
              done = True
              info["successful"] = False

            # else:
            #   theState = self.state.tolist()
            #   within1 = False
            #   for goal in self.goal_states:
            #     if ((theState[0] - 1) <= goal[0] <= (theState[0] + 1)) and ((theState[1] - 1) <= goal[1] <= (theState[1] + 1)):
            #       reward = 1000
            #       within1 = True
            #       break
            #   if not within1:
            #     for goal in self.goal_states:
            #       if ((theState[0] - 2) <= goal[0] <= (theState[0] + 2)) and ((theState[1] - 2) <= goal[1] <= (theState[1] + 2)):
            #         reward = 100
            #         break
              

            if self.img_observation:
                obs = self.state_to_img()
            else:
                obs = self.state_to_oh()
            return obs, reward, done, info


        def render(self, mode='human'):
            """very minimalistic rendering"""
            img = self.state_to_img()
            img = cv2.resize(img, (self.display_size,self.display_size), interpolation=cv2.INTER_NEAREST)
            if mode == 'human':
                fig = plt.figure(0)
                plt.clf()
                plt.imshow(img)
                fig.canvas.draw()
                plt.pause(0.00001)
            return img


        def seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]

    from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
    import numpy as np

    env=GridNavigationEnv()

    env.reset()

    env.close()

    """# 5. Train Model"""

    log_path = os.path.join('Training', 'Logs')

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)

    with suppress_stdout():
      model.learn(total_timesteps=TIMESTEPS_FOR_TRAINING)

    """# 6. Save Model"""

    #model.save('PPO')

    """Evaluate the model"""

    #evaluate_policy(model, env, n_eval_episodes=10, render=True)
    numSuccessful = 0
    episodes = 30
    for episode in range(1, episodes+1):
        #reset the environment and return the initial state
        obs = env.reset()
        done = False
        #keep track of total reward for the episode
        score = 0
        numSteps = 0
        while not done:
            #env.render()
            #get recommended action from model
            action, _states = model.predict(obs)
            action = int(action)
            obs, reward, done, info = env.step(action)
            score+=reward
            numSteps += 1
        if info["successful"] == True:
          numSuccessful += 1
          totalnumSteps += numSteps

    percentSuccessful += numSuccessful / float(30)
    totalSuccessful += numSuccessful
    env.close()

f.write("--------------------------------\n")


f.write(f"average percent successful (APS): {percentSuccessful / float(10)}\n")

f.write(f"average number of steps of episodes that succeeded (ANSES): {float(totalnumSteps) / totalSuccessful}\n")

f.close()
