import numpy as np

from Specs import ModelBase


class Sarsa(ModelBase):
  def __init__(self, state_space_size, action_space_size, alpha=0.01, epsilon=1, decay_rate=0.99, gamma=0.99):
    super().__init__()
    self.alpha = alpha
    self.epsilon = epsilon
    self.gamma = gamma
    self.state_space_size = state_space_size
    self.action_space_size = action_space_size
    self.decay_rate = decay_rate

    # Initialize the Q-table
    self.q_table = np.zeros((state_space_size, action_space_size))

  def choose_action(self, state):
    if np.random.rand() < self.epsilon:
      return np.random.randint(self.action_space_size)  # Explore
    else:
      return np.argmax(self.q_table[state])  # Exploit

  def update(self, state, action, reward, next_state, next_action, done):
    current_q = self.q_table[state, action]
    if done:
      target = reward
    else:
      target = reward + self.gamma * self.q_table[next_state, next_action]

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


class SarsaCV:
  def __init__(self, state_space_size, action_space_size):
    self.state_space_size = state_space_size
    self.action_space_size = action_space_size

  def train_and_evaluate(self, alpha, epsilon, decay_rate, gamma, train_episodes, eval_episodes):
    sarsa = Sarsa(self.state_space_size, self.action_space_size, alpha, epsilon, decay_rate, gamma)
    env = make_random_walk(n_states=19, p_stay=0.0, p_backward=0.5, max_blocks=50)

    # Train the agent
    for episode in range(train_episodes):
      sarsa.decay_epsilon()
      state, _ = env.reset()
      action = sarsa.choose_action(state)
      done = False

      while not done:
        next_state, reward, terminal, truncated, info = env.step(action)
        done = terminal or truncated
        next_action = sarsa.choose_action(next_state)
        sarsa.update(state, action, reward, next_state, next_action, done)
        state = next_state
        action = next_action

    # Evaluate the agent
    success = []
    for _ in range(eval_episodes):
      state, _ = env.reset()
      action = sarsa.predict(state)
      done = False

      while not done:
        next_state, reward, terminal, truncated, info = env.step(action)
        done = terminal or truncated
        action = sarsa.predict(next_state)

      success.append(1 if info.get("success", False) else 0)

    success_rate = np.sum(success) / eval_episodes * 100
    return success_rate


if __name__ == "__main__":

  from itertools import product
  import numpy as np
  import gymnasium as gym
  from Environments.RandomWalk import make_random_walk, estimate_goal_probability
  env = make_random_walk(n_states=19, p_stay=0.0, p_backward=0.5, max_blocks=50)
  estimate_goal_probability(env)

  action_space = env.action_space.n
  state_space = env.observation_space.n

  sarsa_cv = SarsaCV(state_space, action_space)

  # Define hyperparameter grid
  param_grid = {
      "alpha": [0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 1.0],
      "epsilon": [2.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.005, 0.0001],
      "decay_rate": [0.99, 0.995, 0.9, 0.5, 0.1, 0.01],
      "gamma": [0.99, 0.95, 0.9, 0.5, 0.1, 0.01]
  }

  # Train and evaluate SARSA for each combination of hyperparameters
  best_params = None
  best_success_rate = 0
  results = []

  for params in product(*param_grid.values()):
    alpha, epsilon, decay_rate, gamma = params

    success_rate = sarsa_cv.train_and_evaluate(
        alpha, epsilon, decay_rate, gamma,
        train_episodes=10000, eval_episodes=1000
    )

    results.append((alpha, epsilon, decay_rate, gamma, success_rate))
    print(f"Testing params: alpha={alpha}, epsilon={epsilon}, decay_rate={decay_rate}, gamma={gamma}, Success rate: {success_rate:.2f}%")

    if success_rate > best_success_rate:
      best_success_rate = success_rate
      best_params = params

  # Print the best hyperparameters and success rate
  print(f"Best parameters: alpha={best_params[0]}, epsilon={best_params[1]}, "
        f"decay_rate={best_params[2]}, gamma={best_params[3]}")
  print(f"Best success rate: {best_success_rate:.2f}%")
