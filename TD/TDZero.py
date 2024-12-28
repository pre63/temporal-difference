
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from multiprocessing import Pool, get_context
from itertools import product

from CrossValidation import GridSearchCV

from Specs import AlgoSpecs


class TDZero(AlgoSpecs):
  def __init__(self, action_space, observation_space, nrow, ncol, alpha=0.1, gamma=0.99, epsilon=0.1, decay_rate=0.99, policy=None, **kwargs):
    super().__init__()
    self.action_space = action_space
    self.observation_space = observation_space
    self.nrow = nrow
    self.ncol = ncol
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.decay_rate = decay_rate

    self.state_space_size = nrow * ncol
    self.value_table = np.full(self.state_space_size, -np.inf)  # Initialize all states to a very low value

    self.policy = self.egreey_policy if policy is None else policy

  def reset(self):
    self.decay_epsilon()

  def decay_epsilon(self):
    self.epsilon = max(self.epsilon * self.decay_rate, 0.01)

  def egreey_policy(self, state):
    if np.random.rand() < self.epsilon:
      return self.action_space.sample()
    else:
      return self.predict(state)

  def random_policy(self, state):
    return self.action_space.sample()

  def predict(self, state):
    self.value_table[self.value_table == 0] = -np.inf

    if self.action_space.n == 2:
      adjacent_indices = [state - 1, state + 1]
      adjacent_states = [self.value_table[idx] for idx in adjacent_indices if 0 <= idx < self.state_space_size]
      best_action = np.argmax(adjacent_states) if adjacent_states else -1
      return int(best_action)

    if self.action_space.n == 4:
      adjacent_indices = [
          state - 1,              # Left
          state + self.ncol,      # Down
          state + 1,              # Right
          state - self.ncol       # Up
      ]

      # Exclude out-of-bound indices
      adjacent_states = []
      for i, idx in enumerate(adjacent_indices):
        if 0 <= idx < self.state_space_size:
          if (i == 0 and state % self.ncol != 0) or (i == 2 and state % self.ncol != self.ncol - 1) or (i in [1, 3]):
            adjacent_states.append(self.value_table[idx])
          else:
            adjacent_states.append(-np.inf)  # Invalid direction due to grid edges

      best_index = np.argmax(adjacent_states) if adjacent_states else -1
      best_action = [0, 1, 2, 3][best_index] if best_index >= 0 else -1

      return int(best_action)

  def update(self, **kwargs):
    state = kwargs.get("state")
    reward = kwargs.get("reward")
    next_state = kwargs.get("next_state")
    done = kwargs.get("done")
    self._update(state, reward, next_state, done)

  def _update(self, state, reward, next_state, done):
    current_value = 0 if self.value_table[state] == -np.inf else self.value_table[state]
    next_value = 0 if self.value_table[next_state] == -np.inf else self.value_table[next_state]

    td_target = reward + (0 if done else self.gamma * next_value)
    td_error = td_target - current_value
    new_value = self.alpha * td_error

    assert not np.isinf(new_value)
    assert not np.isnan(new_value)

    self.value_table[state] = current_value + new_value
    self.value_table[self.value_table == 0] = -np.inf


class TDZeroCV(GridSearchCV):
  def new(self, **params):
    return TDZero(
        action_space=self.env.action_space,
        observation_space=self.env.observation_space,
        nrow=self.env.nrow,
        ncol=self.env.ncol,
        **params
    )


if __name__ == "__main__":
  from Environments.RandomWalk import make_random_walk, estimate_goal_probability
  from Environments.FrozenLake import make_frozen_lake

  env = make_random_walk()  # Best results: alpha=0.003, gamma=0.7
  env = make_frozen_lake()  # Best results: alpha=0.003, gamma=0.2

  param_grid = {
      "alpha": np.linspace(0.0001, 1.0, 50),
      "gamma": np.linspace(0.0001, 1.0, 50),
  }
  param_grid = {
      "alpha": [0.003],
      "gamma": [0.2],
  }

  cv = TDZeroCV(env, param_grid)
  cv.search()

  estimate_goal_probability(env)
  cv.summary()
