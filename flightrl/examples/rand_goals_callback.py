#!/usr/bin/env python3

'''
@script description

'''

# builtin 
import copy
import pdb
import math

# third-party
import numpy as np
import torch 
import cv2

from stable_baselines3.common.callbacks import BaseCallback
from scipy.spatial.transform import Rotation 


# own mods

__author__ = "Linn of the Ded"
__status__ = "Production"




class RandGoalsCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, waypt_gap_m=10.0, goal_tol_m=3.0, verbose=0):
    # def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super(RandGoalsCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        print("Setting rand goals for training...")
        self.waypt_gap_m=waypt_gap_m # max distance between the gen goal n the robot at time of gen
        self.goal_tol_m=goal_tol_m # min dist to consider the goal is met
        self.goal=np.array([5.0, 5.0, 5.0], dtype=np.float32)
        self.is_reached_goal=False # TODO: this is not used but is set correctly
        self.rollout_id=0
        self.count_ts =0

        # TODO check if cfg file is used correctly, move to init just once
        cam_inQ_pos=np.array([0.176, 0.0, 0.05])
        cam_inQ_euler_rot=np.array( [-90.0, 0.0, 5.0]) #yaw pitch roll (ZYX)
        cam_inQ_mat_rot = Rotation.from_euler('zyx', cam_inQ_euler_rot, degrees=True)
        # hetero
        cam_inQ_het =  cam_inQ_mat_rot.as_matrix()
        cam_inQ_het=np.append(cam_inQ_het, np.array([[0,0,0]]), axis=0)
        self.cam_inQ_het=np.append(cam_inQ_het.T, np.array([np.append(cam_inQ_pos,[1.0])]), axis=0).T

        # FOV of quad TODO: check if these r correct cuz i just cp paste fr the c source files n idk if they r the final ones
        self.frame_height = 128
        self.frame_width = 128
        self.camera_FOV = 90

        self.focal_len = (self.frame_height / 2.0) * math.tan(math.pi*self.camera_FOV/(180.*2.))


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """

        self.rollout_id +=1
        #if self.rollout_id % 300 ==0 :
        if True:
            print("300 rollouts, new goal")

            """
            offset=np.random.uniform(self.waypt_gap_m*-1, self.waypt_gap_m, size=(3))

            # if it is too close to curr pose
            while np.absolute(offset[0]) < 1.0 or np.absolute(offset[1]) < 1.0:
                # TODO: I could make this more efficient but I dun feel like it...
                offset=np.random.uniform(self.waypt_gap_m*-1, self.waypt_gap_m, size=(3))

            # TODO: make goal offset fr curr pos. rmb to check bounds
            new_goal=offset
            """

            new_goal=np.zeros((3))
            new_goal[0]=1.176
            new_goal[1]=0.0
            new_goal[2]=4.0

            print("new goal ", new_goal)
    
            self.goal=np.array(new_goal, dtype=np.float32)
            self.training_env.set_goal(self.goal)
            self.is_reached_goal=False

            new_start=np.zeros((3))
            # new_start[2]=np.random.uniform(4.0, 6.0)
            new_start[2]=4.0
            self.training_env.set_resetpos(np.array(new_start, dtype=np.float32))


    def is_in_fov(self, quad_inW_pos, quad_inW_euler_rot,  goal_inW_pos):
        # camera intrinsics
        quad_inW_mat_rot = Rotation.from_euler('zyx', quad_inW_euler_rot, degrees=True)
        #hetero TODO make into fn u lazy a**
        quad_inW_het=quad_inW_mat_rot.as_matrix()
        quad_inW_het=np.append(quad_inW_het, np.array([[0,0,0]]), axis=0)
        quad_inW_het=np.append(quad_inW_het.T, np.array([np.append(quad_inW_pos,[1.0])]), axis=0).T

        # transform. darn i rly should hav just found a lib for dis bt i was too lazy
        cam_inW_het = np.matmul(quad_inW_het, self.cam_inQ_het)

        cam_inW_mat_rot=cam_inW_het[0:3,0:3]

        goal_inC_pos = np.matmul(cam_inW_mat_rot.T, goal_inW_pos) - np.matmul(cam_inW_mat_rot.T, cam_inW_het[:3,-1])

        goal_inC_theta_xy=math.atan(goal_inC_pos[0]/goal_inC_pos[1]) # radians
        goal_inC_theta_yz=math.atan(goal_inC_pos[2]/goal_inC_pos[1]) # radians

        is_fov= abs(goal_inC_theta_xy) > (math.pi*self.camera_FOV/(180.*2.)) or abs(goal_inC_theta_yz) > (math.pi*self.camera_FOV/(180.*2.))
        is_fov = not is_fov

        print("goal pos in W: ", goal_inW_pos)
        print("goal pos in C: ", goal_inC_pos)
        print("cam in W: ", cam_inW_het)
        print("counter", self.count_ts)

        return is_fov, goal_inC_theta_xy, goal_inC_theta_yz, goal_inC_pos


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        
        # print("global keys", self.globals.keys())
        # print("local keys", self.locals.keys())

        quad_inW_pos=self.locals["new_obs"][0,:3]
        quad_inW_euler_rot=self.locals["new_obs"][0,3:6] # yaw pitch roll cuz the ori authors are a sadistic bunch
        goal_inW_pos=(self.locals["new_obs"])[0,12:]

        is_fov, goal_inC_theta_xy, goal_inC_theta_yz, goal_inC_pos= self.is_in_fov(quad_inW_pos, quad_inW_euler_rot,  goal_inW_pos)

        # extract images
        img=self.training_env.get_images()

        # FIXME move to DBG only: paint dot on image to test goal pos
        if is_fov:
            row = self.focal_len * goal_inC_pos[0] / goal_inC_pos[1]
            row = int(row + (128/2))
            col = self.focal_len * goal_inC_pos[2] / goal_inC_pos[1]
            col = int(col + (128/2))
            # pdb.set_trace()
            img[0,row,col]=254.
            cv2.imwrite("./test_imgs/test"+str(self.count_ts)+".png", img[0])
        else:
            print("goal not in fov")
        self.count_ts +=1
        
        obs_tensor=(self.locals["new_obs"])
        # obs_tensor=(self.locals["obs_tensor"]).numpy()
        # print("obs: ", self.locals["new_obs"])

        # check if near goal
        pos=obs_tensor[0,:3]
        dist=np.sqrt(np.sum((self.goal-pos) ** 2))

        if dist < self.goal_tol_m: # if reached goal
            self.is_reached_goal=True 

        # diff and normalise before giving PPO
        new_obs=copy.deepcopy(obs_tensor)
        new_obs[0,0]=obs_tensor[0,0]-obs_tensor[0,12]
        new_obs[0,1]=obs_tensor[0,1]-obs_tensor[0,13]
        new_obs[0,2]=obs_tensor[0,2]-obs_tensor[0,14]
        new_obs[0,12]=0.0
        new_obs[0,13]=0.0
        new_obs[0,14]=0.0
        # new_obs=new_obs/10.0
        self.locals["new_obs"]=new_obs
        self.locals["obs_tensor"]=torch.from_numpy(new_obs)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

