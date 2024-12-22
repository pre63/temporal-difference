
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

  def _evaluate_model(self, model, trials=100):
    success, rewards = 0, 0

    for _ in range(trials):
      state, _ = self.env.reset()
      done = False

      while not done:
        action = model.predict(state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        state = next_state

      rewards += reward
      success += 1 if info.get("success", False) else 0

    avg_success = (success / trials) * 100
    avg_rewards = rewards / trials

    return avg_success, avg_rewards

  def _evaluate_params(self, args):
    params, episodes, samples = args
    all_data = []

    for _ in range(samples):
      model = self.new(**params)
      sample_data = self._train_and_collect_data(model, episodes)
      all_data.append(sample_data)

    avg_success, avg_rewards = self._evaluate_model(model, episodes // 10)

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
      model.reset()
      episode_rewards = []
      done = False

      while not done:
        action = model.policy(state)
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

  def plot_metrics(self, index=None):
    if index is not None:
      if index < 0 or index >= len(self.results):
        print(f"Invalid index: {index}. Please provide a valid index.")
        return
      result = self.results[index]
    else:
      top_5 = sorted(self.results, key=lambda x: x['avg_success'], reverse=True)[:5]
      if len(top_5) == 0:
        print("No results to plot")
        return
      result = top_5[0]

    samples = result['samples']

    plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2)

    # Plot rewards per episode for all samples
    ax1 = plt.subplot(gs[0, 0])
    for j, sample in enumerate(samples):
      ax1.plot(sample['rewards'], label=f"Sample {j + 1}")
    ax1.set_title(f"Rewards per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Rewards")
    ax1.legend()

    # Plot steps per episode for all samples
    ax2 = plt.subplot(gs[0, 1])
    for j, sample in enumerate(samples):
      ax2.plot(sample['steps'], label=f"Sample {j + 1}")
    ax2.set_title(f"Steps per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.legend()

    # Plot value function for all samples as a line plot
    ax3 = plt.subplot(gs[1, :])
    for j, sample in enumerate(samples):
      ax3.plot(sample['value_function'], label=f"Sample {j + 1}")
    ax3.set_title(f"Value Function")
    ax3.set_xlabel("State Index")
    ax3.set_ylabel("Predicted Value")
    ax3.legend()

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
  param_grid = {
      "alpha": [0.002],
      "gamma": [0.001],
  }

  cv = TDZeroCV(env, param_grid)
  cv.search()

  estimate_goal_probability(env)
  cv.summary()
  cv.plot_metrics()
