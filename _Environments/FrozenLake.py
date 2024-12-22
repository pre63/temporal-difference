import gymnasium as gym
# import timelimit warapper
from gymnasium.wrappers import TimeLimit


class FrozenLakeWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.name = "FrozenLake-v1"
    self.step_counter = 0
    self.observation = None
    self.nrow = self.env.unwrapped.nrow
    self.ncol = self.env.unwrapped.ncol

  def reset(self):
    self.step_counter = 0
    self.observation = None
    return self.env.reset()

  def step(self, action):

    state, reward, terminated, truncated, info = self.env.step(action)

    # Add success info to info dict
    info["success"] = reward == 1

    # Penalty for not earning reward
    if reward == 0:
      reward = -0.001

    # Penalty for falling into the hole
    if terminated and state != 15:
      reward = -1

    # Penalty for not moving
    if self.observation is not None and self.observation == state:
      reward = -1

    self.observation = state
    self.step_counter += 1

    return state, reward, terminated, truncated, info


def make_frozen_lake(render_mode=None):
  env = gym.make("FrozenLake-v1", is_slippery=False, render_mode=render_mode)
  env = TimeLimit(env, max_episode_steps=100)
  return FrozenLakeWrapper(env)
