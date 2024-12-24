import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import product

from Specs import ModelBase


class ExpectedSarsa(ModelBase):
  def __init__(self, state_space_size, action_space_size, alpha=0.01, epsilon=1.0, decay_rate=0.99, gamma=0.99):
    self.alpha = alpha
    self.epsilon = epsilon
    self.gamma = gamma
    self.state_space_size = state_space_size
    self.action_space_size = action_space_size
    self.decay_rate = decay_rate

    # Initialize the Q-table
    self.q_table = np.zeros((state_space_size, action_space_size))

  def policy(self, state):
    if np.random.rand() < self.epsilon:
      return np.random.randint(self.action_space_size)  # Explore
    else:
      return np.argmax(self.q_table[state])  # Exploit

  def update(self, **kwargs):
    state = kwargs.get("state")
    action = kwargs.get("action")
    reward = kwargs.get("reward")
    next_state = kwargs.get("next_state")
    done = kwargs.get("done")
    self._update(state, action, reward, next_state, done)

  def _update(self, state, action, reward, next_state, done):
    current_q = self.q_table[state, action]

    if done:
      target = reward
    else:
      # Calculate the expected value of the next state
      next_action_probabilities = np.ones(self.action_space_size) * (self.epsilon / self.action_space_size)
      best_action = np.argmax(self.q_table[next_state])
      next_action_probabilities[best_action] += (1.0 - self.epsilon)
      expected_q = np.dot(next_action_probabilities, self.q_table[next_state])

      target = reward + self.gamma * expected_q

    self.q_table[state, action] += self.alpha * (target - current_q)

  def decay_epsilon(self):
    self.epsilon = max(self.epsilon * self.decay_rate, 0.01)

  def predict(self, state):
    return np.argmax(self.q_table[state])

  def save(self, filename):
    np.savez(filename, q_table=self.q_table, params={"alpha": self.alpha, "epsilon": self.epsilon, "gamma": self.gamma})

  def load(self, filename):
    data = np.load(filename + ".npz", allow_pickle=True)
    self.q_table = data["q_table"]
    params = data["params"].item()
    self.alpha = params["alpha"]
    self.epsilon = params["epsilon"]
    self.gamma = params["gamma"]


from CrossValidation import GridSearchCV


class ExpectedSarsaCV(GridSearchCV):
  def new(self, **params):
    return ExpectedSarsa(
        state_space_size=self.env.observation_space.n,
        action_space_size=self.env.action_space.n,
        **params
    )


if __name__ == "__main__":
  from Environments.RandomWalk import make_random_walk, estimate_goal_probability
  from Environments.FrozenLake import make_frozen_lake

  env = make_frozen_lake()  # Best alpha=0.01, epsilon=2.0, decay_rate=0.99, gamma=0.99
  env = make_random_walk()  # alpha=0.01, epsilon=2.0, decay_rate=0.99, gamma=0.95

  param_grid = {
      "alpha": [0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1.0],
      "epsilon": [2.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.005, 0.0001],
      "decay_rate": [0.99, 0.995, 0.9, 0.5, 0.1, 0.01],
      "gamma": [0.99, 0.95, 0.9, 0.5, 0.1, 0.01]
  }

  cv = ExpectedSarsaCV(env, param_grid)
  cv.search()

  estimate_goal_probability(env)
  cv.summary()
  cv.plot_metrics()
