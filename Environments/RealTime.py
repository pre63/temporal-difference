import gym
import numpy as np
from datetime import datetime


class RealTimeInformationWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.total_data_volume = 0  # Total data processed (bytes)
    self.total_duration = 0  # Total real-time duration (seconds)
    self.episode_data_volume = 0  # Data volume for the current episode
    self.episode_duration = 0  # Real-time duration for the current episode
    self.start_time = None  # Tracks start of real-time measurement

  def _calculate_data_size(self, obj):
    """
    Calculate the size of an object in bytes.
    Supports numpy arrays, integers, and floats.
    """
    if isinstance(obj, np.ndarray):
      return obj.nbytes
    elif isinstance(obj, (int, float)):
      return 8  # Assuming 8 bytes for Python floats or integers
    else:
      raise ValueError(f"Unsupported data type for size calculation: {type(obj)}")

  def reset(self, **kwargs):
    # Reset episode-specific counters
    self.episode_data_volume = 0
    self.episode_duration = 0
    self.start_time = datetime.now()  # Start timing for the new episode
    obs = super().reset(**kwargs)

    # Add the size of the initial observation to total and episode data volume
    observation_size = self._calculate_data_size(obs)
    self.total_data_volume += observation_size
    self.episode_data_volume += observation_size

    return obs

  def step(self, action):
    # Capture the current time for elapsed time calculation
    current_time = datetime.now()

    # Calculate real-time elapsed duration
    if self.start_time is not None:
      elapsed_time = (current_time - self.start_time).total_seconds()
      self.episode_duration += elapsed_time
      self.total_duration += elapsed_time

    # Update start_time for the next step
    self.start_time = current_time

    # Take a step in the environment
    obs, reward, done, info = super().step(action)

    # Calculate data sizes for the current observation, action, and reward
    action_size = self._calculate_data_size(action)
    observation_size = self._calculate_data_size(obs)
    reward_size = self._calculate_data_size(reward)

    # Accumulate data volume
    current_data_volume = action_size + observation_size + reward_size
    self.total_data_volume += current_data_volume
    self.episode_data_volume += current_data_volume

    # Calculate real-time bandwidth
    self.current_bandwidth = current_data_volume / elapsed_time if elapsed_time > 0 else 0
    self.episode_bandwidth = self.episode_data_volume / max(1e-6, self.episode_duration)
    self.total_bandwidth = self.total_data_volume / max(1e-6, self.total_duration)

    return obs, reward, done, info

  def get_metrics(self):
    return {
        "current_bandwidth": self.current_bandwidth,
        "episode_bandwidth": self.episode_bandwidth,
        "total_bandwidth": self.total_bandwidth,
        "total_data_volume": self.total_data_volume,
        "episode_data_volume": self.episode_data_volume,
        "total_duration": self.total_duration,
        "episode_duration": self.episode_duration
    }
