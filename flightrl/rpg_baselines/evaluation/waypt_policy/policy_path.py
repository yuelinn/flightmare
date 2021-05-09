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
# from stable_baselines3.sac.sac import SAC

class ObstacleAvoidanceAgent():
  def __init__(self, 
               num_envs : int = 1,
               num_acts : int = 4,
               ):
    self._num_envs = num_envs
    self._num_acts = num_acts
    
    # initialization
    weigthts_path="/root/flightmare/flightrl/examples/saved/2021-05-07-13-29-53.zip" # f20d882db6de7df91fe1fbbef58beea6214520b9
    
    self._model = PPO.load(weigthts_path)
    # self._model = SAC.load(weigthts_path)
        

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

    # preprocessing
    # TODO: make preprocessing a callable function
    obs[0,-3:]= current_goal_position
    obs[0,0]=obs[0,0]-obs[0,12]
    obs[0,1]=obs[0,1]-obs[0,13]
    obs[0,2]=obs[0,2]-obs[0,14]
    obs[0,12]=0.0
    obs[0,13]=0.0
    obs[0,14]=0.0
    obs=obs/10.0
    act, _ = self._model.predict(obs, deterministic=True)
    action=act
    # print(action)
    # action = np.zeros([self._num_envs,self._num_acts], dtype=np.float32)
    # action[0,0] += -0.01
    return action
