
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from multiprocessing import Pool, get_context
from itertools import product


class GridSearchCV:
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
    pass

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

    # Train multiple models with the given parameters
    for _ in range(samples):
      model = self.new(**params)
      sample_data = self._train_and_collect_data(model, episodes)
      sample_data['model'] = model
      all_data.append(sample_data)

    # Select the best model based on the highest sum of successes
    best_sample = max(all_data, key=lambda x: sum(x['rewards']))
    best_model = best_sample['model']
    avg_success, avg_rewards = self._evaluate_model(best_model, episodes // 20)

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
      episode_rewards = []
      done = False

      model.reset()

      state, _ = self.env.reset()
      action = model.policy(state)

      while not done:
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_action = model.policy(next_state)
        model.update(state, action, reward, next_state, next_action, done)
        done = terminated or truncated

        episode_rewards.append(reward)
        state = next_state
        action = next_action

      rewards.append(sum(episode_rewards))
      steps.append(len(episode_rewards))
      success.append(0 if info.get("success", False) else 1)

    value_function = self._generate_value_function(model)

    return {'rewards': rewards, 'steps': steps, 'success': sum(success), 'value_function': value_function}

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
