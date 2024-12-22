import numpy as np
from collections import defaultdict


class TemporalDifference:
  def __init__(self, n_states, n_actions, alpha, gamma, epsilon):
    self.q_table = defaultdict(lambda: np.zeros(n_actions))
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.n_actions = n_actions

  def choose_action(self, state):
    if np.random.random() < self.epsilon:
      return np.random.randint(self.n_actions)
    return np.argmax(self.q_table[state])

  def update(self, state, action, reward, next_state, next_action, done):
    td_target = reward + (0 if done else self.gamma * self.q_table[next_state][next_action])
    td_error = td_target - self.q_table[state][action]
    self.q_table[state][action] += self.alpha * td_error

  def predict(self, state):
    return np.argmax(self.q_table[state])


if __name__ == "__main__":
  import gymnasium as gym
  from Environments.RandomWalk import make_random_walk

  env = make_random_walk(n_states=19, p_stay=0.0, p_backward=0.1, max_blocks=10)

  # Initialize critic and actor-critic models
  action_space = env.action_space.n
  state_space = env.observation_space.n
  td = TemporalDifference(state_space, action_space, alpha=0.1, gamma=0.99, epsilon=0.1)

  # Train the agent
  episodes = 1000
  for episode in range(episodes):
    if episode % (episodes // 10) == 0 and episode > 0:
      print(f"Episode: {episode}")
    state, _ = env.reset()
    action = td.choose_action(state)
    done = False
    while not done:
      next_state, reward, terminal, truncated, info = env.step(action)
      done = terminal or truncated
      next_action = td.choose_action(next_state)
      td.update(state, action, reward, next_state, next_action, done)
      state = next_state
      action = next_action

  # Evaluate the agent 100 episode print the %
  success = []
  for episode in range(100):
    state, _ = env.reset()
    action = td.predict(state)
    done = False

    print(f"Episode: {episode}")

    while not done:
      next_state, reward, terminal, truncated, info = env.step(action)
      print(f"State: {state}, Action: {action}, Reward: {reward}", info)
      done = terminal or truncated
      next_action = td.predict(next_state)
      state = next_state
      action = next_action

    success.append(1 if info.get("success", False) else 0)

  print(f"Eval Success Rate: { np.sum(success) / len(success) * 100 }%")
