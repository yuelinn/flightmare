import numpy as np
from gym import spaces
from stable_baselines3.common.vec_env import VecEnv
import cv2 as cv


class FlightEnvVec(VecEnv):
    #
    def __init__(self, impl):
        self.wrapper = impl
        self.num_obs = self.wrapper.getObsDim()
        self.num_acts = self.wrapper.getActDim()
        self.frame_dim = self.wrapper.getFrameDim()
        print("Observations: ", self.num_obs)
        print("Actions: ", self.num_acts)
        print("image shape:", self.frame_dim)
        self._observation_space = spaces.Box(low=-np.ones(self.num_obs) * np.inf, high=np.ones(self.num_obs) * np.inf, dtype=np.float32)
        self._action_space = spaces.Box(
            low=np.ones(self.num_acts) * -1.,
            high=np.ones(self.num_acts) * 1.,
            dtype=np.float32)
        self._observation = np.zeros((self.num_envs, self.num_obs), dtype=np.float32)
        self._images = np.zeros((self.num_envs, self.frame_dim[0], self.frame_dim[1]), dtype=np.float32)
        self._odometry = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self.img_array = np.zeros((self.num_envs, self.frame_dim[0]*self.frame_dim[1]), dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool)
        self._extraInfoNames = self.wrapper.getExtraInfoNames()
        self._extraInfo = np.zeros([self.num_envs,
                                    len(self._extraInfoNames)], dtype=np.float32)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.count = 0

        self.max_episode_steps = 300

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def set_goal(self, goal):
        # print("vec env wrapper: setting new goal: ", goal)
        self.wrapper.set_goal(goal)
        return True

    def set_resetpos(self, goal):
        # print("vec env wrapper: setting new goal: ", goal)
        self.wrapper.set_resetpos(goal)
        return True
        
    def set_objects_densities(self, object_density_fractions):
        if(self.render):
            self.wrapper.setObjectsDensities(object_density_fractions)
            print("density set to ", object_density_fractions)
        return

    def obs_array2image(self):
        return self.img_array.reshape((self.num_envs, self.frame_dim[0], self.frame_dim[1]), order='F')

    def step(self, action):
        self.wrapper.step(action, self._odometry, self.img_array,
                          self._reward, self._done, self._extraInfo)
        
        self._observation = self._odometry

        # Images are accessible from here, self._images is of shape [self.num_envs, self.frame_dim[0], self.frame_dim[1]]
        self._images = self.obs_array2image()

        # ----- Uncomment below to check if images are correct -----
        # if self.count < 100:
        #     cv.imwrite('images/img'+str(self.count)+'.png', self._images[0,:,:]/15*255)
        #     self.count = self.count + 1

        if len(self._extraInfoNames) is not 0:
            info = [{'extra_info': {
                self._extraInfoNames[j]: self._extraInfo[i, j] for j in range(0, len(self._extraInfoNames))
            }} for i in range(self.num_envs)]
        else:
            info = [{} for i in range(self.num_envs)]

        for i in range(self.num_envs):
            self.rewards[i].append(self._reward[i])
            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                info[i]['episode'] = epinfo
                self.rewards[i].clear()

        # The observations returned are only the one related to the odometry NO images
        # you have to change the observation space in order to use them

        return self._observation.copy(), self._reward.copy(), \
            self._done.copy(), info.copy()
    
    def get_images(self):
        return self._images.copy()


    def stepUnity(self, action, send_id):
        receive_id = self.wrapper.stepUnity(action, self._odometry, self.img_array,
                                            self._reward, self._done, self._extraInfo, send_id)

        return receive_id

    def sample_actions(self):
        actions = []
        for i in range(self.num_envs):
            action = self.action_space.sample().tolist()
            actions.append(action)
        return np.asarray(actions, dtype=np.float32)

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset(self._odometry, self.img_array)
        self.obs_array2image()
        return self._observation.copy()

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()
        return info

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

    def connectUnity(self):
        return self.wrapper.connectUnity()


    def disconnectUnity(self):
        self.wrapper.disconnectUnity()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def extra_info_names(self):
        return self._extraInfoNames

    def start_recording_video(self, file_name):
        raise RuntimeError('This method is not implemented')

    def stop_recording_video(self):
        raise RuntimeError('This method is not implemented')

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def step_async(self):
        raise RuntimeError('This method is not implemented')

    def step_wait(self):
        raise RuntimeError('This method is not implemented')

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError('This method is not implemented')

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError('This method is not implemented')

    def env_is_wrapped(self):
        """
        Check if environments are wrapped with a given wrapper.
        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: True if the env is wrapped, False otherwise, for each env queried.
        """
        return np.ones([self.num_envs], dtype=bool).tolist()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError('This method is not implemented')
