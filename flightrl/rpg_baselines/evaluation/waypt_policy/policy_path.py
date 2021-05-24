#!/usr/bin/env python3

"""
Hover policy that came with the box
Does not take in images
Cannot set waypoints
"""

import pdb
import math

import numpy as np
from typing import List
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC


class ObstacleAvoidanceAgent():
  def __init__(self, 
               num_envs : int = 1,
               num_acts : int = 4,
               ):
    self._num_envs = num_envs
    self._num_acts = num_acts
    
    # initialization
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-03-21-08-58.zip" # hover
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-06-18-52-12.zip" #lyfe sucks
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-05-17-08-02.zip" # 25mil
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-07-15-49-30.zip" # 292805d
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-07-18-34-52/weights/w_time__3400000_steps.zip" # 25mil+3mil
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-07-13-29-53.zip" # f20d882db6de7df91fe1fbbef58beea6214520b9
    # weigthts_path="/root/flightmare/flightrl/examples/saved/w_time__3400000_steps.zip" # f20d882db6de7df91fe1fbbef58beea6214520b9
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-08-17-44-39.zip" # 40mil
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-14-15-39-06.zip" # hover 0,0,x no term
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-15-18-34-06.zip" # hover 0,0,x no term
    weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-22-20-17-44.zip" # idgaf
    self._model = PPO.load(weigthts_path)


    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-10-15-47-24.zip" # SAC 6mil
    # self._model = SAC.load(weigthts_path)
        


  def get_local(self, global_goal, curr_pos, tol=2.0):
    # interpolate in straight line global to curr goal
    delta=global_goal-curr_pos
    dist=np.absolute(delta)
    is_big=dist>tol

    for it, is_big_x in enumerate(is_big):
      if is_big_x:
        if delta[it] > tol:
          delta[it] = tol
        else:
          delta[it] = -1.0* tol

    local_goal= curr_pos + delta


    # TODO: avoid extreme heights. since the goal is met if it is near, avoid going too low or too high (risk locations)
    if local_goal[-1] < 1.5:
      local_goal[-1] = 1.5
    if local_goal[-1] > 7.0:
      local_goal[-1] =7.0  

    # local_goal[-1]=global_goal[-1]
    print("global goal, curr pos, local goal:", global_goal, curr_pos, local_goal)
    return local_goal

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

    local_waypt = self.get_local(current_goal_position[0:3], obs[0,0:3])

    obs[0,-3:]= local_waypt
    obs[0,0]=obs[0,0]-obs[0,12]
    obs[0,1]=obs[0,1]-obs[0,13]
    obs[0,2]=obs[0,2]-obs[0,14]+5.0
    obs[0,12]=0.0
    obs[0,13]=0.0
    obs[0,14]=0.0

    # FIXME try this
    # obs[0,12]=0.0
    # obs[0,13]=0.0
    # obs[0,14]=5.0

    # FIXME hover no div of 10
    # obs=obs/10.0

    act, _ = self._model.predict(obs, deterministic=True)
    action=act
    # print(action)
    # action = np.zeros([self._num_envs,self._num_acts], dtype=np.float32)
    # action[0,0] += -0.01
    return action
