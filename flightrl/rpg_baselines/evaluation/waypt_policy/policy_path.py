#!/usr/bin/env python3

"""
Hover policy that came with the box
Does not take in images
Cannot set waypoints
"""

import pdb
import math
import copy
from colorama import Fore, Back, Style

import numpy as np
from typing import List
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.sac.sac import SAC
from scipy.spatial.transform import Rotation 

import sys
sys.path.insert(0, '../../examples')
from rand_goals_callback import RandGoalsCallback


class ObstacleAvoidanceAgent():
  def __init__(self, 
               num_envs : int = 1,
               num_acts : int = 4,
               ):
    self._num_envs = num_envs
    self._num_acts = num_acts
    
    # initialization
    weigthts_path="/root/flightmare/flightrl/examples/saved/2021-05-03-21-08-58.zip" # hover

    self._model = PPO.load(weigthts_path)

    self.callback_obj=RandGoalsCallback()

    self.z_error=0.
    self.r_error=0.
    self.p_error=0.
    self.y_error=0.
    self.posx_error=0.
    self.posy_error=0.
    self.i_r = 0.
    self.i_p = 0.
    self.i_posx=0
    self.i_posy=0


  def get_local(self, global_goal, curr_pos, curr_ori, img, tol=2.0):
    is_move=False

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

    # avoid extreme heights. since the goal is met if it is near, avoid going too low (aerodynamics issue)
    min_h=2.0
    if local_goal[-1] < min_h:
      local_goal[-1] = min_h

    # check if path to goal is clear
    quad_inW_pos=curr_pos
    quad_inW_euler_rot=curr_ori
    goal_inW_pos=copy.deepcopy(local_goal)
    des_rot=copy.deepcopy(curr_ori) # YPR
    is_collide=TrueW
    siam = 0.5
    iterations = 0
    max_iter = 8 # 2m in each direction

    while is_collide:
      # take turns to check left and right
      coeff = iterations/2
      if iterations%2 ==0:
        coeff=coeff*-1.0
      goal_inW_pos[0]=local_goal[0]+coeff*siam

      iterations +=1

      is_fov, goal_inC_theta_xy, goal_inC_theta_yz, goal_inC_pos = self.callback_obj.is_in_fov(quad_inW_pos, quad_inW_euler_rot,  goal_inW_pos)

      if is_fov:
        is_collide = self.callback_obj.is_goal_collide(img, goal_inC_pos)
        # if is_collide:
          # print(Fore.RED+"obstacle found. trying to evade....!!!!")
          # print(Style.RESET_ALL)
          # pdb.set_trace()
      else:
        break

    local_goal=goal_inW_pos

    # print("global goal, curr pos, local goal:", global_goal, curr_pos, local_goal)
    return local_goal, is_move


  """
  obs: (np.array (1,15)) curr pose 
  done: (bool) if True then stop
  images: (np array (1, 128, 128))images
  current_goal_position: (np array (3,)) goal from high level planner

  Returns 
    actions: np array float32 (num_envs, num_acts)
  """
  def getActions(self, obs, done, images, current_goal_position):

    local_waypt, is_move = self.get_local(current_goal_position[0:3], obs[0,0:3], obs[0,3:6], images)

    # transform to goal frame
    obs[0,-3:]= local_waypt
    obs[0,0]=obs[0,0]-obs[0,12]
    obs[0,1]=obs[0,1]-obs[0,13]
    obs[0,2]=obs[0,2]-obs[0,14]+5.0
    obs[0,12]=0.0
    obs[0,13]=0.0
    obs[0,14]=0.0

    act, _ = self._model.predict(obs, deterministic=True)

    action=act

    return action
