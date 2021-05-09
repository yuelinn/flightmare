#!/usr/bin/env python3

'''
@script description

'''

# builtin 

# third-party
import torch as th

# own mods

__author__ = "Linn of the Ded"
__status__ = "Production"

from stable_baselines3.common.callbacks import BaseCallback


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
        if self.rollout_id % 300 ==0 : # FIXME: check wy SAC keeps calling this at every episode
            print("300 rollouts, new goal")

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
            self.is_reached_goal=False

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        
        # print("global keys", self.globals.keys())
        # print("local keys", self.locals.keys())

        
        obs_tensor=(self.locals["new_obs"])
        # obs_tensor=(self.locals["obs_tensor"]).numpy()
        print("obs: ", self.locals["new_obs"])

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
        new_obs=new_obs/10.0

        # get images
        img=copy.deepcopy(self.training_env.get_images()) # np arr of shape (1, 128, 128)
        img=img/255.0

        new_obs=np.expand_dims(np.append(img, new_obs),0)
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

