import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import product


class ExpectedSarsa:
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

  def update(self, state, action, reward, next_state, done):
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


class ExpectedSarsaCV:
  def __init__(self, env, param_grid):
    self.env = env
    self.param_grid = param_grid
    self.state_space_size = env.observation_space.n
    self.action_space_size = env.action_space.n
    self.results = []
    self.episode_rewards = []
    self.episode_steps = []
    self.value_functions = []

  def search(self, episodes=10000):
    for params in product(*self.param_grid.values()):
      alpha, epsilon, decay_rate, gamma = params

      success_rate, rewards, steps, value_function = self.train_and_evaluate(
          alpha=alpha,
          epsilon=epsilon,
          decay_rate=decay_rate,
          gamma=gamma,
          episodes=episodes,
          eval_episodes=episodes // 10
      )

      self.results.append({
          'params': {
              'alpha': alpha,
              'epsilon': epsilon,
              'decay_rate': decay_rate,
              'gamma': gamma
          },
          'avg_success': success_rate,
          'samples': {
              'rewards': rewards,
              'steps': steps,
              'value_function': value_function
          }
      })

      self.episode_rewards.append(rewards)
      self.episode_steps.append(steps)
      self.value_functions.append(value_function)

      print(f"Testing params: alpha={alpha}, epsilon={epsilon}, decay_rate={decay_rate}, gamma={gamma}, "
            f"Success rate: {success_rate:.2f}%")

  def summary(self):
    top_5 = sorted(self.results, key=lambda x: x['avg_success'], reverse=True)[:5]
    print("Top 5 results:")
    for result in top_5:
      params = result['params']
      success_rate = result['avg_success']
      print(f"alpha={params['alpha']}, epsilon={params['epsilon']}, decay_rate={params['decay_rate']}, "
            f"gamma={params['gamma']}, success_rate={success_rate:.2f}%")

  def train_and_evaluate(self, alpha, epsilon, decay_rate, gamma, episodes, eval_episodes):
    expected_sarsa = ExpectedSarsa(
        self.state_space_size,
        self.action_space_size,
        alpha=alpha,
        epsilon=epsilon,
        decay_rate=decay_rate,
        gamma=gamma
    )

    rewards = []
    steps = []

    # Train the agent
    for episode in range(episodes):
      expected_sarsa.decay_epsilon()
      state, _ = self.env.reset()
      episode_rewards = 0
      episode_steps = 0
      done = False

      while not done:
        action = expected_sarsa.policy(state)
        next_state, reward, terminal, truncated, info = self.env.step(action)
        done = terminal or truncated
        expected_sarsa.update(state, action, reward, next_state, done)
        state = next_state

        episode_rewards += reward
        episode_steps += 1

      rewards.append(episode_rewards)
      steps.append(episode_steps)

    # Generate the value function
    value_function = self._generate_value_function(expected_sarsa)

    # Evaluate the agent
    success = []
    for _ in range(eval_episodes):
      state, _ = self.env.reset()
      done = False

      while not done:
        action = expected_sarsa.predict(state)
        next_state, reward, terminal, truncated, info = self.env.step(action)
        done = terminal or truncated
        state = next_state

      success.append(1 if info.get("success", False) else 0)

    success_rate = np.sum(success) / eval_episodes * 100
    return success_rate, rewards, steps, value_function

  def _generate_value_function(self, model):
    value_function = []
    for state in range(self.state_space_size):
      value = model.predict(state)
      value_function.append(value)
    return value_function

  def plot_metrics(self, index=None):
    if index is not None:
      if index < 0 or index >= len(self.results):
        print(f"Invalid index: {index}. Please provide a valid index.")
        return
      result = self.results[index]
    else:
      result = max(self.results, key=lambda x: x['avg_success'])

    samples = result['samples']

    plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2)

    # Plot rewards per episode
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(samples['rewards'], label="Rewards")
    ax1.set_title("Rewards per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Rewards")
    ax1.legend()

    # Plot steps per episode
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(samples['steps'], label="Steps")
    ax2.set_title("Steps per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.legend()

    # Plot value function as a line plot
    ax3 = plt.subplot(gs[1, :])
    ax3.plot(samples['value_function'], label="Value Function")
    ax3.set_title("Value Function")
    ax3.set_xlabel("State Index")
    ax3.set_ylabel("Predicted Value")
    ax3.legend()

    plt.tight_layout()
    plt.show()


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
