
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
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
      left, right = state - 1, state + 1

      if state == 0:
        adjacent_states = self.value_table[right]
      if state == self.state_space_size - 1:
        adjacent_states = self.value_table[left]
      if state > 0 and state < self.state_space_size - 1:
        adjacent_states = self.value_table[[left, right]]

      best_action = np.argmax(adjacent_states)
      return int(best_action)

    if self.action_space.n == 4:
      # 0: Move left
      # 1: Move down
      # 2: Move right
      # 3: Move up
      left, down, right, up = state - 1, state + self.ncol, state + 1, state - self.ncol

      if state % self.ncol == 0:
        adjacent_states = self.value_table[[down, right, up]]
      if state % self.ncol == self.ncol - 1:
        adjacent_states = self.value_table[[left, down, up]]
      if state < self.ncol:
        adjacent_states = self.value_table[[left, down, right]]
      if state >= self.nrow * (self.ncol - 1):
        adjacent_states = self.value_table[[left, right, up]]
      if state % self.ncol != 0 and state % self.ncol != self.ncol - 1 and state >= self.ncol and state < self.nrow * (self.ncol - 1):
        adjacent_states = self.value_table[[left, down, right, up]]

      best_index = np.argmax(adjacent_states)
      best_action = [0, 1, 2, 3][best_index]

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

    with Pool(16) as pool:
      self.results = pool.map(
          self._evaluate_params,
          [(params, episodes, samples) for params in param_combinations]
      )

  def _evaluate_params(self, args):
    params, episodes, samples = args
    success_rates = []

    for _ in range(samples):
      model = self.new(**params)
      success_rate, rewards, steps, value_func = self.train_and_evaluate(model, episodes)
      success_rates.append(success_rate)

      self.episode_rewards.append(rewards)
      self.episode_steps.append(steps)
      self.value_function.append(value_func)

    avg_success_rate = np.mean(success_rates)
    alpha, gamma = params["alpha"], params["gamma"]
    print(f"Alpha: {alpha:.4f}, Gamma: {gamma:.4f}, Average Success: {avg_success_rate:.2f}%")

    return params, avg_success_rate

  def train_and_evaluate(self, model, episodes):
    rewards = []
    steps = []
    for _ in range(episodes):
      state, _ = self.env.reset()
      total_reward = 0
      step_count = 0
      done = False

      while not done:
        action = self.env.action_space.sample()
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        model.update(state, reward, next_state, done)
        state = next_state
        total_reward += reward
        step_count += 1

      rewards.append(total_reward)
      steps.append(step_count)

    # Generate value function using the model's prediction across the state space
    value_func = self._generate_value_function(model)

    success = []
    for _ in range(episodes // 10):
      state, _ = self.env.reset()
      done = False

      while not done:
        action = model.predict(state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        state = next_state

      success.append(1 if info["success"] else 0)

    success_rate = np.mean(success) * 100
    return success_rate, rewards, steps, value_func

  def _generate_value_function(self, model):
    """Generate the value function for visualization using the model's prediction."""
    value_func = []
    for state in range(self.env.observation_space.n):
      value = model.predict(state)
      value_func.append(value)
    return value_func

  def summary(self):
    top_5 = sorted(self.results, key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 results:")
    for i, (params, success_rate) in enumerate(top_5):
      rewards_per_ep = np.mean(self.episode_rewards[i])
      steps_per_ep = np.mean(self.episode_steps[i])
      print(f"Run {i + 1}: params={params}, success_rate={success_rate:.2f}%, "
            f"avg_rewards_per_episode={rewards_per_ep:.2f}, avg_steps_per_episode={steps_per_ep:.2f}")

  def plot_metrics(self):
    for i, (rewards, steps, value_func) in enumerate(zip(self.episode_rewards, self.episode_steps, self.value_function)):
      fig = plt.figure(figsize=(16, 8))
      gs = GridSpec(2, 2, figure=fig)

      # Plot rewards per episode
      ax1 = fig.add_subplot(gs[0, 0])
      ax1.plot(rewards)
      ax1.set_title(f"Rewards per Episode - Run {i + 1}")
      ax1.set_xlabel("Episode")
      ax1.set_ylabel("Total Reward")

      # Plot steps per episode
      ax2 = fig.add_subplot(gs[0, 1])
      ax2.plot(steps)
      ax2.set_title(f"Steps per Episode - Run {i + 1}")
      ax2.set_xlabel("Episode")
      ax2.set_ylabel("Steps")

      # Plot value function
      ax3 = fig.add_subplot(gs[1, :])
      im = ax3.imshow(value_func, cmap="viridis", interpolation="nearest")
      fig.colorbar(im, ax=ax3)
      ax3.set_title(f"Value Function - Run {i + 1}")
      ax3.set_xlabel("Grid Column")
      ax3.set_ylabel("Grid Row")

      plt.tight_layout()
      plt.show()


if __name__ == "__main__":
  from Environments.RandomWalk import make_random_walk, estimate_goal_probability
  from Environments.FrozenLake import make_frozen_lake

  env = make_frozen_lake()  # Best results: alpha=0.003, gamma=0.2
  env = make_random_walk()  # Best results: alpha=0.003, gamma=0.7

  param_grid = {
      "alpha": np.linspace(0.0001, 1.0, 50),
      "gamma": np.linspace(0.0001, 1.0, 50),
  }

  cv = TDZeroCV(env, param_grid)
  cv.search()

  estimate_goal_probability(env)
  cv.summary()
