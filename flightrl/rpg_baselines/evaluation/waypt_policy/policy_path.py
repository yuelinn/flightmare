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


# copied from https://learnopencv.com/rotation-matrix-to-euler-angles/ cuz im lazy
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([z, y, x])



class ObstacleAvoidanceAgent():
  def __init__(self, 
               num_envs : int = 1,
               num_acts : int = 4,
               ):
    self._num_envs = num_envs
    self._num_acts = num_acts
    
    # initialization
    weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-03-21-08-58.zip" # hover
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-06-18-52-12.zip" #lyfe sucks
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-05-17-08-02.zip" # 25mil
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-07-15-49-30.zip" # 292805d
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-07-18-34-52/weights/w_time__3400000_steps.zip" # 25mil+3mil
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-07-13-29-53.zip" # f20d882db6de7df91fe1fbbef58beea6214520b9
    # weigthts_path="/root/flightmare/flightrl/examples/saved/w_time__3400000_steps.zip" # f20d882db6de7df91fe1fbbef58beea6214520b9
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-08-17-44-39.zip" # 40mil
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-14-15-39-06.zip" # hover 0,0,x no term
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-15-18-34-06.zip" # hover 0,0,x no term
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-26-20-31-42/weights/w_time__5810000_steps.zip" # idgaf
    self._model = PPO.load(weigthts_path)

    self.callback_obj=RandGoalsCallback()
    # weigthts_path="/root/challenge/flightrl/examples/saved/2021-05-10-15-47-24.zip" # SAC 6mil
    # self._model = SAC.load(weigthts_path)


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

    # TODO: avoid extreme heights. since the goal is met if it is near, avoid going too low or too high (risk locations)
    min_h=2.0
    if local_goal[-1] < min_h:
      local_goal[-1] = min_h
    # if local_goal[-1] > 7.0:
    #   local_goal[-1] =7.0  

    # local_goal[-1]=global_goal[-1]

    # check if path to goal is clear
    quad_inW_pos=curr_pos
    quad_inW_euler_rot=curr_ori
    goal_inW_pos=copy.deepcopy(local_goal)
    des_rot=copy.deepcopy(curr_ori) # YPR
    is_collide=True
    siam = 0.5
    iterations = 0
    max_iter = 8 # 2m in each direction

    # while False: #FIXME testing stuff
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
        if is_collide:
          print(Fore.RED+"obstacle found. trying to evade....!!!!")
          print(Style.RESET_ALL)
          # pdb.set_trace()
      else:
        # move to see the goal
        '''
        if goal_inC_theta_xy > self.callback_obj.camera_FOV/2:
          print("i should move to the right")
          is_move=True
        else:
          is_move=True
          print("i should move to the left")
        print("bt i aint cuz i cant cuz controls still cant rotate... so just try to get to the goal for now")
        '''
        print("obs out of fov. ignore checks")
        break

    local_goal=goal_inW_pos

    print("global goal, curr pos, local goal:", global_goal, curr_pos, local_goal)
    return local_goal, is_move


  def rotate_me(self, local_waypt, q_pos, q_rot):

    quad_inW_mat_rot = Rotation.from_euler('zyx', q_rot, degrees=False)
    quad_inW_het=quad_inW_mat_rot.as_matrix()
    quad_inW_het=np.append(quad_inW_het, np.array([[0,0,0]]), axis=0)
    quad_inW_het=np.append(quad_inW_het.T, np.array([np.append(q_pos,[1.0])]), axis=0).T

    des_yaw=math.atan((q_pos[1]-local_waypt[1] ) /(local_waypt[0] - q_pos[0]))
    # max_yaw=0.01745329 # 1deg
    max_yaw=0.01745329*5 # 5deg
    # max_yaw=0.0000000001
    # if abs(des_yaw) > max_yaw:
    #   des_yaw = des_yaw/abs(des_yaw)*max_yaw

    des_yaw=q_rot[0]+0.01# FIXME DBG
    gp_inW_rot = Rotation.from_euler('zyx', np.array([des_yaw,0.,0.]), degrees=False)
    gp_inW_het=gp_inW_rot.as_matrix()
    gp_inW_het=np.append(gp_inW_het, np.array([[0,0,0]]), axis=0)
    gp_inW_het=np.append(gp_inW_het.T, np.array([np.append(local_waypt,[1.0])]), axis=0).T

    gp_inG=np.eye(4)
    gp_inG[2,-1]=5.

    w_inG=np.matmul(gp_inG, np.linalg.inv(gp_inW_het))

    uav_inG=np.matmul(w_inG, quad_inW_het)

    new_obs=uav_inG[0:3,-1]
    new_obs[0]=-1.0
    new_obs[1]=-1.0
    new_obs[2]=5.0

    des_euler = rotationMatrixToEulerAngles(uav_inG[:3,:3])

    # FIXME DBG
    # if not np.array_equal(q_rot, des_euler):
    #   print("should b equal",q_rot, des_euler)
      # pdb.set_trace()

    return new_obs, des_euler  


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

    local_waypt, is_move = self.get_local(current_goal_position[0:3], obs[0,0:3], obs[0,3:6], images)
    obs[0,-3:]= local_waypt
    obs[0,0]=obs[0,0]-obs[0,12]
    obs[0,1]=obs[0,1]-obs[0,13]
    obs[0,2]=obs[0,2]-obs[0,14]+5.0
    obs[0,12]=0.0
    obs[0,13]=0.0
    obs[0,14]=0.0

    # fix rotation to point front of uav to the goal dir
    # local_waypt, des_euler = self.rotate_me(local_waypt, obs[0,0:3], obs[0,3:6])

    # obs[0,0]=local_waypt[0]
    # obs[0,1]=local_waypt[1]
    # obs[0,2]=local_waypt[2]
    # obs[0,12]=0.0
    # obs[0,13]=0.0
    # obs[0,14]=0.0
    # # print("should b equal",obs[0,3:6], des_euler)
    # # new rotations
    # obs[0,3]= des_euler[0] # radians y
    # obs[0,4]= des_euler[1] # radians p
    # obs[0,5]= des_euler[2] # radians r

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
