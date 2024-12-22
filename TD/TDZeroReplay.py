
import numpy as np


from algorithms.TDZero import TDZero, TDZeroCV

epsilon = 1e-10


class TDZeroReplay(TDZero):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    state_space_size = self.state_space_size
    self.eligibility_matrix = np.identity(state_space_size)

  def update(self, state, reward, next_state, done):
    # Handle initialization for the first visit
    if self.value_table[state] == -np.inf:
      self.value_table[state] = 0
    if next_state < self.state_space_size and self.value_table[next_state] == -np.inf:
      self.value_table[next_state] = 0

    # Compute TD error
    current_value = self.value_table[state]
    next_value = 0 if done else self.value_table[next_state]
    td_error = reward + self.gamma * next_value - current_value

    # Update eligibility matrix
    phi = np.zeros(self.state_space_size)
    phi[state] = 1
    phi_next = np.zeros(self.state_space_size)
    if not done and next_state < self.state_space_size:
      phi_next[next_state] = 1

    self.eligibility_matrix = self.gamma * self.eligibility_matrix + np.outer(phi, phi_next)

    # Perform full replay update
    replay_update = td_error * np.dot(self.eligibility_matrix, phi)

    # Apply updates to visited states only
    visited_indices = np.where(self.value_table != -np.inf)[0]
    self.value_table[visited_indices] += self.alpha * replay_update[visited_indices]

    # Keep unvisited states as -inf
    self.value_table[self.value_table == 0] = -np.inf


class TDZeroReplayCV(TDZeroCV):

  def new(self, **params):
    return TDZeroReplay(
        action_space=self.env.action_space,
        observation_space=self.env.observation_space,
        nrow=self.env.nrow,
        ncol=self.env.ncol,
        **params
    )


if __name__ == "__main__":
  from Environments.RandomWalk import make_random_walk, estimate_goal_probability
  from Environments.FrozenLake import make_frozen_lake

  env = make_random_walk()  # alpha=0.005, gamma=1.0, success_rate=40.10%
  env = make_frozen_lake()  # alpha=0.002, gamma=0.6,  success_rate=90.00%

  param_grid = {
      "alpha": [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005],
      "gamma": [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
  }
  param_grid = {
      "alpha": np.linspace(0.0001, 1.0, 50),
      "gamma": np.linspace(0.0001, 1.0, 50),
  }

  cvr = TDZeroReplayCV(env, param_grid)
  cvr.search()

  estimate_goal_probability(env)
  cvr.summary()
