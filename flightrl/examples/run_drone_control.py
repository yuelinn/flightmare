#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

#
import os
import math
import argparse
import numpy as np
import time
import sys
import torch
import pdb
import copy

#
from stable_baselines3.common import logger

#
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback

from rpg_baselines.ppo.ppo2_test import test_model
import rpg_baselines.common.util as U
from rpg_baselines.envs import vec_env_wrapper as wrapper
from scipy.spatial.transform import Rotation 
#
from flightgym import QuadrotorEnv_v1


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
        self.is_reached_goal=False

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

        # if(self.is_reached_goal):
        if True:
            print("new episode, new goal")

            offset=np.random.uniform(self.waypt_gap_m*-1, self.waypt_gap_m, size=(3))

            # if it is too close to curr pose
            while np.absolute(offset[0]) < 1.0 or np.absolute(offset[1]) < 1.0:
                # TODO: I could make this more efficient but I dun feel like it...
                offset=np.random.uniform(self.waypt_gap_m*-1, self.waypt_gap_m, size=(3))

            # TODO: make goal offset fr curr pos. rmb to check bounds
            new_goal=offset

            # check bounds
            new_goal[2]=np.random.uniform(0.1, 8.0)

            print("new goal ", new_goal)

            self.goal=np.array(new_goal, dtype=np.float32)

            self.training_env.set_goal(self.goal)


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        
        # print("global keys", self.globals.keys())
        # print("local keys", self.locals.keys())

        # check if near goal
        obs_tensor=(self.locals["new_obs"])
        print("obs: ", self.locals["new_obs"])

        # obs_tensor=(self.locals["obs_tensor"]).numpy()
        pos=obs_tensor[0,:3]
        # goal=obs_tensor[0,-3:]
        dist=np.sqrt(np.sum((self.goal-pos) ** 2))

        if dist < self.goal_tol_m: # if reached goal
            self.is_reached_goal=True # set a new goal

        # diff and normalise before giving PPO
        new_obs=copy.deepcopy(obs_tensor)
        new_obs[0,0]=obs_tensor[0,0]-obs_tensor[0,12]
        new_obs[0,1]=obs_tensor[0,1]-obs_tensor[0,13]
        new_obs[0,2]=obs_tensor[0,2]-obs_tensor[0,14]
        new_obs[0,12]=0.0
        new_obs[0,13]=0.0
        new_obs[0,14]=0.0
        new_obs=new_obs/10.0
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




def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=1,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('-w', '--weight', type=str, default='./saved/quadrotor_env.zip',
                        help='trained weight path')
    return parser


def main():
    args = parser().parse_args()
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    env = wrapper.FlightEnvVec(QuadrotorEnv_v1(
        dump(cfg, Dumper=RoundTripDumper), False))
    if args.render:
        connectedToUnity = False 
        while not connectedToUnity:
            connectedToUnity = env.connectUnity()             
            if not connectedToUnity:  
                print("Couldn't connect to unity, will try another time.")
    
    print("env.num_envs : ", env.num_envs)

    max_ep_length = env.max_episode_steps

    # FIXME: no obstacles for now
    object_density_fractions = np.zeros([env.num_envs], dtype=np.float32)
    # if env.num_envs == 1:
    #     object_density_fractions = np.ones([env.num_envs], dtype=np.float32)
    # else:
    #     object_density_fractions = np.linspace(0.0, 1.0, num=env.num_envs)

    env.set_objects_densities(object_density_fractions = object_density_fractions)   
    env.reset()
    
    time.sleep(2.5)    
    # set random seed
    configure_random_seed(args.seed, env=env)

    #
    if args.train:
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved'
        saver = U.ConfigurationSaver(log_dir=log_dir)
        # model = PPO('MlpPolicy', env, verbose=2, tensorboard_log=saver.data_dir)

        model=PPO.load("./saved/2021-05-03-20-34-51.zip") # fine tuning from hover policy
        model.set_env(env)

        # tensorboard
        # Make sure that your chrome browser is already on.
        # TensorboardLauncher(saver.data_dir + '/PPO2_1')

        # PPO run
        # Originally the total timestep is 5 x 10^8
        # 10 zeros for nupdates to be 4000
        # 1000000000 is 2000 iterations and so
        # 2000000000 is 4000 iterations.
        logger.configure(folder=saver.data_dir)
        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=saver.data_dir+'/weights/', name_prefix='w_time_')
        # TODO: make callback that saves only if the returns have improved

        randgoalscallback=RandGoalsCallback()

        model.learn(
            total_timesteps=int(3000000), callback=[randgoalscallback, checkpoint_callback], tb_log_name="test")
        model.save(saver.data_dir)

    # # Testing mode with a trained weight
    else:
        model = PPO.load(args.weight)
        test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()
