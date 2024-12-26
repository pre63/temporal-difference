import numpy as np
from itertools import product
from scipy.sparse import csr_matrix

from Sarsa.IHT import IHT, tiles
from Specs import ModelBase


class SarsaContinuous(ModelBase):
  def __init__(self, observation_space, action_space, alpha=0.01, epsilon=0.1, gamma=1, tilings=20, max_episode_steps=10000, action_resolution=3):
    super().__init__()
    self.alpha = alpha
    self.epsilon = epsilon
    self.gamma = gamma
    self.tilings = tilings
    self.max_episode_steps = max_episode_steps

    self.observation_space = observation_space
    self.action_space = action_space

    self.action_space_low, self.action_space_high = action_space.low, action_space.high

    self.action_values = np.array(
        list(product(
            *[np.linspace(low, high, action_resolution) for low, high in zip(self.action_space_low, self.action_space_high)]
        ))
    )

    iht_size = int(self.tilings * 4 ** len(self.observation_space.shape))
    self.w = np.random.uniform(low=-0.05, high=0.05, size=(iht_size,))
    print(f"Size of weights: {len(self.w)}")

    self.tile_coding = IHT(iht_size)

  def q_(self, feature):
    return feature.dot(self.w).sum()

  def update(self, **params):
    state = params.get("state")
    action = params.get("action")
    reward = params.get("reward")
    done = params.get("done")
    q = params.get("q")
    next_q = params.get("next_q")

    self._update(reward, q, next_q, state, action, done)

  def _update(self, reward, q, next_q, state, action, done):
    feature = self._hash(state, action).flatten()  # Convert sparse matrix to dense array
    feature_norm = np.linalg.norm(feature)
    if feature_norm == 0:
      feature_norm = 1  # Avoid division by zero

    td_error = reward + (self.gamma * next_q if not done else 0) - q
    self.w += (self.alpha * td_error / feature_norm) * feature

  def _one_hot_encode(self, indices):
    size = len(self.w)
    one_hot_vec = np.zeros(size)
    one_hot_vec[indices] = 1
    return one_hot_vec

  def _hash(self, state, action):
    feature_ind = np.array(tiles(self.tile_coding, self.tilings, state.tolist(), action.tolist()))
    feature = self._one_hot_encode(feature_ind)
    return feature

  def policy(self, state):
    action, q = self._choose_action(state)
    return action, q

  def _choose_action(self, state, deterministic=False):
    action_val_dict = {}
    for action in self.action_values:
      feature = self._hash(state, action)
      q = self.q_(feature)
      action_val_dict[tuple(action)] = q

    greedy_action = max(action_val_dict, key=action_val_dict.get)

    if deterministic:
      return np.array(greedy_action), action_val_dict[tuple(greedy_action)]

    non_greedy_actions = [np.array(a) for a in set(action_val_dict.keys()) - {greedy_action}]
    prob_explorative_action = self.epsilon / len(self.action_values)
    prob_greedy_action = 1 - self.epsilon + prob_explorative_action

    actions = [np.array(greedy_action)] + non_greedy_actions
    probabilities = [prob_greedy_action] + [prob_explorative_action] * len(non_greedy_actions)

    chosen_action_index = np.random.choice(range(len(actions)), p=probabilities)
    chosen_action = actions[chosen_action_index]

    return chosen_action, action_val_dict[tuple(chosen_action)]

  def decay_epsilon(self):
    self.epsilon = max(self.epsilon - (1.0 / self.max_episode_steps), 0)

  def predict(self, state):
    action, q = self._choose_action(state, deterministic=True)
    action = np.clip(action, self.action_space_low, self.action_space_high)
    action = action.flatten()
    return np.array(action), q


if __name__ == "__main__":
  from Environments.LunarLander import make_lunar_lander
  import numpy as np

  def evaluate(env, model):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
      action, _ = model.predict(state)
      next_state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      state = next_state
      total_reward += reward
    success = 1 if info.get("Success", False) else 0
    return success, total_reward

  env = make_lunar_lander()

  sarsa = SarsaContinuous(
      action_space=env.action_space,
      observation_space=env.observation_space,
      alpha=0.003,
      gamma=0.99,
  )

  batch_size = 100
  max_episodes = 10000

  for batch in range(max_episodes // batch_size):
    success_count = 0
    batch_rewards = []

    for episode in range(batch_size):
      state, _ = env.reset()
      sarsa.decay_epsilon()
      sarsa.reset()
      done = False
      total_reward = 0
      q = None

      while not done:
        action, q = sarsa.policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_action, next_q = sarsa.policy(next_state)
        done = terminated or truncated
        sarsa.update(state=state, action=action, reward=reward, done=done, q=q, next_q=next_q)
        state = next_state
        action = next_action
        q = next_q
        total_reward += reward

      batch_rewards.append(total_reward)

    success_rate = []
    for _ in range(100):
      success, total_reward = evaluate(env, sarsa)
      print(f"Success: {success}, Total Reward: {total_reward}")
      success_rate.append(success)

    success_rate = sum(success_rate) / len(success_rate)

    print(f"Batch {batch + 1}/{max_episodes // batch_size}: Success Rate = {success_rate * 100:.2f}%")

  env.close()
  print("Training complete.")

  # # Plot success rates
  # import matplotlib.pyplot as plt
  # plt.plot(range(1, len(success_rates) + 1), success_rates)
  # plt.xlabel("Batch Number")
  # plt.ylabel("Success Rate")
  # plt.title("SARSA Success Rate Evolution")
  # plt.show()

  # Render a trained agent

  env = make_lunar_lander(render_mode="human")
  evaluate(env, sarsa)
