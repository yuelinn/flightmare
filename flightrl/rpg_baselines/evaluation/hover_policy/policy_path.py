#!/usr/bin/env python3

"""
Hover policy that came with the box
Does not take in images
Cannot set waypoints
"""

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
    # TODO: call RL model
    weigthts_path="/root/challenge/flightrl/examples/saved/2021-04-02-16-58-04.zip"
    self._model = PPO.load(weigthts_path)
        

  """
  obs: (np.array (1,15)) curr pose 
  done: (bool) if True then stop
  images: (np array (1, 128, 128))images
  current_goal_position: (np array (3,)) goal from high level planner

  Returns 
    actions: np array float32 (num_envs, num_acts)
  """
  def getActions(self, obs, done, images, current_goal_position):
    # override current_goal_position
    # current_goal_position=np.zeros([3,], dtype=np.float32)

    act, _ = self._model.predict(obs, deterministic=True)
    action=act
    # print(action)
    # action = np.zeros([self._num_envs,self._num_acts], dtype=np.float32)
    # action[0,0] += -0.01
    # TODO call RL model
    return action
