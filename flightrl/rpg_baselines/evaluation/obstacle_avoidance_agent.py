#!/usr/bin/env python3

import numpy as np
import math
from typing import List

from stable_baselines3.ppo.ppo import PPO

class ObstacleAvoidanceAgent():
  def __init__(self, 
               num_envs : int = 1,
               num_acts : int = 4,
               ):
    self._num_envs = num_envs
    self._num_acts = num_acts
    
    # initialization
        
  def getActions(self, obs, done, images, current_goal_position):
    action = np.zeros([self._num_envs,self._num_acts], dtype=np.float32)
    action[0,0] += -0.01
    return action

"""
  @brief: Wrapper for base class using the PPO algorithm 
  @param: load model weights from zip file (as per the run_drone_control.py
  method) and instantiate PPO model
  @note: training is performed with the goal state set to 0, hence the 
  actual "state" passed to model = drone_state - current_goal 
  for all environments 
"""
class PPOAgent(ObstacleAvoidanceAgent):
  def __init__(self, 
              weights : None,  
              num_envs : int = 1, 
              num_acts : int = 4): 
      self.weights = weights
      self.num_envs = num_envs 
      self.num_acts = num_acts
      self.model = PPO.load(self.weights, deterministic=True)

      if(self.weights is None): 
        print("No weights passed to model")
      else:
        PPO.load(self.weights)
  
  def getActions(self, obs, done, images, current_goal_position): 
    # substraction should be for all environments
    obs[:,0:3] =  np.reshape(current_goal_position, obs[:, 0:3].shape) - obs[:,0:3]
    obs[:,12:] = np.zeros(obs[:,12:].shape)
    print("model state " + str(obs))
    act, _ = self.model.predict(obs, deterministic=True)
    return act

