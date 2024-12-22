
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool, get_context
from itertools import product


class TDZero:
  def __init__(self, action_space, observation_space, nrow, ncol, alpha=0.1, gamma=0.99, epsilon=0.1, decay_rate=0.99, policy=None, **kwargs):

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

    self.policy = self.random_policy if policy is None else policy

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

  def update(self, state, reward, next_state, done):
    current_value = 0 if self.value_table[state] == -np.inf else self.value_table[state]
    next_value = 0 if self.value_table[next_state] == -np.inf else self.value_table[next_state]

    td_target = reward + (0 if done else self.gamma * next_value)
    td_error = td_target - current_value
    new_value = self.alpha * td_error

    assert not np.isinf(new_value)
    assert not np.isnan(new_value)

    self.value_table[state] = current_value + new_value
    self.value_table[self.value_table == 0] = -np.inf


class TDZeroCV:
  def __init__(self, env, param_grid):
    self.env = env
    self.param_grid = param_grid
    self.results = []
    self.episode_rewards = []
    self.episode_steps = []
    self.value_function = []

    print("Number of permutations:", len(param_grid["alpha"]) * len(param_grid["gamma"]))
    print("Alpha values:", [float(f"{x:.4f}") for x in param_grid['alpha']])
    print("Gamma values:", [float(f"{x:.4f}") for x in param_grid['gamma']])

  def new(self, **params):
    return TDZero(
        action_space=self.env.action_space,
        observation_space=self.env.observation_space,
        nrow=self.env.nrow,
        ncol=self.env.ncol,
        **params
    )

  def search(self, episodes=4000, samples=5):
    param_combinations = [
        dict(zip(self.param_grid.keys(), values))
        for values in product(*self.param_grid.values())
    ]

    # Use 'spawn' to avoid print suppression
    with get_context("spawn").Pool() as pool:
      self.results = pool.map(
          self._evaluate_params,
          [(params, episodes, samples) for params in param_combinations]
      )

  def _evaluate_params(self, args):
    params, episodes, samples = args
    all_data = []

    for _ in range(samples):
      model = self.new(**params)
      sample_data = self._train_and_collect_data(model, episodes)
      all_data.append(sample_data)

    avg_success = np.mean([np.mean(sample['success']) for sample in all_data])
    avg_rewards = np.mean([np.mean(sample['rewards']) for sample in all_data])

    result = {
        'params': params,
        'avg_success': avg_success,
        'avg_rewards': avg_rewards,
        'samples': all_data
    }

    print(f"Alpha: {params['alpha']:.4f}, Gamma: {params['gamma']:.4f}, Avg Success: {avg_success:.2f}%, Avg Rewards: {avg_rewards:.2f}")

    return result

  def _train_and_collect_data(self, model, episodes):
    rewards, steps, success = [], [], []

    for _ in range(episodes):
      state, _ = self.env.reset()
      episode_rewards = []
      done = False

      while not done:
        action = model.predict(state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        model.update(state, reward, next_state, done)
        done = terminated or truncated

        episode_rewards.append(reward)
        state = next_state

      rewards.append(sum(episode_rewards))
      steps.append(len(episode_rewards))
      success.append(0 if info.get("success", False) else 1)

    value_function = self._generate_value_function(model)

    return {'rewards': rewards, 'steps': steps, 'success': success, 'value_function': value_function}

  def _generate_value_function(self, model):
    """Generate the value function for visualization using the model's prediction."""
    value_func = []
    for state in range(self.env.observation_space.start, self.env.observation_space.n):
      value = model.predict(state)
      if self.env.observation_space.contains(value):
        value_func.append(value)
      else:
        value_func.append(None)  # Handle out-of-bounds predictions
    return value_func

  def summary(self):
    top_5 = sorted(self.results, key=lambda x: x['avg_success'], reverse=True)[:5]
    print("Top 5 Results:")
    for i, result in enumerate(top_5):
      print(f"Run {i + 1}: {result['params']}, Avg Success: {result['avg_success']:.2f}%, Avg Rewards: {result['avg_rewards']:.2f}")

  def plot_metrics(self):
    top_5 = sorted(self.results, key=lambda x: x['avg_success'], reverse=True)[:5]

    for i, result in enumerate(top_5):
      rewards = [r for sample in result['samples'] for r in sample['rewards']]
      steps = [s for sample in result['samples'] for s in sample['steps']]

      plt.figure(figsize=(12, 8))
      gs = GridSpec(2, 2)

      # Plot rewards per episode
      ax1 = plt.subplot(gs[0, 0])
      ax1.plot(rewards)
      ax1.set_title(f"Rewards per Episode - Top {i + 1}")
      ax1.set_xlabel("Episode")
      ax1.set_ylabel("Total Rewards")

      # Plot steps per episode
      ax2 = plt.subplot(gs[0, 1])
      ax2.plot(steps)
      ax2.set_title(f"Steps per Episode - Top {i + 1}")
      ax2.set_xlabel("Episode")
      ax2.set_ylabel("Steps")

      # Plot value function
      ax3 = plt.subplot(gs[1, :])
      value_func = result['samples'][0]['value_function']
      im = ax3.imshow(value_func, cmap="viridis", interpolation="nearest")
      plt.colorbar(im, ax=ax3)
      ax3.set_title(f"Value Function - Top {i + 1}")
      ax3.set_xlabel("Grid Column")
      ax3.set_ylabel("Grid Row")

      plt.tight_layout()
      plt.show()


if __name__ == "__main__":
  from Environments.RandomWalk import make_random_walk, estimate_goal_probability
  from Environments.FrozenLake import make_frozen_lake

  env = make_random_walk()  # Best results: alpha=0.003, gamma=0.7
  env = make_frozen_lake()  # Best results: alpha=0.003, gamma=0.2

  param_grid = {
      "alpha": np.linspace(0.0001, 1.0, 50),
      "gamma": np.linspace(0.0001, 1.0, 50),
  }

  cv = TDZeroCV(env, param_grid)
  cv.search()

  estimate_goal_probability(env)
  cv.summary()
  cv.plot_metrics()
