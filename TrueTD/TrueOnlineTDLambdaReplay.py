import numpy as np
from itertools import product


# Scale features
def binary_basis_features(n_obs, n_actions):
  def feature_func(state, action=None):
    if action is None:
      action = 0
    feature = np.zeros(n_obs * n_actions)
    feature[state * n_actions + action] = 1
    return feature / np.sqrt(n_obs)  # Normalize features
  return feature_func


class TrueOnlineTDLambdaReplay:
  def __init__(self, alpha, gamma, lambd, theta_init, n_obs, n_actions, action_space, feature_func=None):
    self.alpha = alpha
    self.gamma = gamma
    self.lambd = lambd
    self.theta = theta_init.copy()
    self.feature_func = feature_func if feature_func is not None else binary_basis_features(n_obs, n_actions)
    self.n = len(self.theta)
    self.n_obs = n_obs
    self.n_actions = n_actions
    self.action_space = action_space

    # Initialization of traces and auxiliary variables
    self.e = np.zeros(self.n)  # Eligibility trace
    self.e_bar = np.zeros(self.n)  # Auxiliary eligibility trace
    self.A_bar = np.eye(self.n)  # Auxiliary matrix (identity matrix)
    self.V_old = 0.0  # Previous value
    self.phi = None

    assert np.all(np.isfinite(self.theta)), "Theta contains invalid values."
    assert np.all(np.isfinite(self.A_bar)), "A_bar contains invalid values."
    assert np.all(np.isfinite(self.e)), "Eligibility trace contains invalid values."
    assert np.all(np.isfinite(self.e_bar)), "Auxiliary trace contains invalid values."

  def reset(self):
    """Resets traces and auxiliary variables at the start of an episode."""
    self.e = np.zeros(self.n)
    self.e_bar = np.zeros(self.n)
    self.A_bar = np.eye(self.n)
    self.V_old = 0.0
    self.phi = None

  def reset_traces(self):
    """Resets only the eligibility traces for within-episode updates."""
    self.e = np.zeros(self.n)
    self.e_bar = np.zeros(self.n)
    self.V_old = 0.0
    self.phi = None

  def update(self, state, action, reward, next_state, next_action, done):
    if self.phi is None:
      self.phi = self.feature_func(state, action)

    phi_next = self.feature_func(next_state, next_action) if not done else np.zeros(self.n)
    V = np.dot(self.theta, self.phi)
    V_next = np.dot(self.theta, phi_next)
    delta = reward + (0 if done else self.gamma * V_next) - V

    # Update eligibility trace
    trace_cap = 10
    self.e = np.clip(self.gamma * self.lambd * self.e + (1 - self.alpha * self.gamma * self.lambd * np.dot(self.e, self.phi)) * self.phi, -trace_cap, trace_cap)

    # Update auxiliary eligibility trace
    self.e_bar = np.clip(
        self.e_bar + self.e * (delta + V - self.V_old) - self.alpha * self.phi * (np.dot(self.e_bar, self.phi) - self.V_old),
        -trace_cap, trace_cap
    )

    # Update auxiliary matrix with stabilization
    self.A_bar = (1 - 1e-4) * self.A_bar - self.alpha * np.outer(self.phi, np.dot(self.phi, self.A_bar)) + 1e-6 * np.eye(self.n)

    # Update parameter vector with regularization
    gradient = self.A_bar @ self.theta + self.e_bar
    gradient_norm = np.linalg.norm(gradient)
    max_gradient_norm = 1.0
    if gradient_norm > max_gradient_norm:
      gradient *= max_gradient_norm / gradient_norm

    self.theta = self.theta - self.alpha * gradient - 1e-3 * self.theta  # Regularization

    # Update for the next iteration
    self.V_old = V_next
    self.phi = None if done else phi_next

    return self.theta

  def predict(self, state, action):
    features = self.feature_func(state, action)  # Compute feature representation for state
    pred = np.dot(self.theta, features)  # Dot product for prediction
    pred = np.clip(pred, -1e3, 1e3)  # Clip the prediction to avoid extreme values

    if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
      # Continuous action space
      action = np.clip(pred, self.action_space.low, self.action_space.high)
    elif hasattr(self.action_space, 'n'):
      # Discrete action space
      action = int(np.round(pred))  # Round prediction to nearest integer
      action = np.clip(action, 0, self.action_space.n - 1)  # Clip to valid discrete range
    else:
      raise ValueError("Unknown action space type")

    return action


class TrueOnlineTDLambdaReplayCV:

  def __init__(self, env, param_grid):
    self.env = env
    self.param_grid = param_grid
    self.state_space_size = env.observation_space.n
    self.action_space_size = env.action_space.n
    self.results = []

  def search(self):
    # Train and evaluate True Online TD(lambda) for each combination of hyperparameters

    print("Searching hyperparameters...")
    print(f"Total combinations: {np.prod([len(v) for v in self.param_grid.values()])}")

    for params in product(*self.param_grid.values()):
      alpha, gamma, lambd = params
      print(f"Training with alpha={alpha}, gamma={gamma}, lambd={lambd}...")

      success_rate = self.train_and_evaluate(
          alpha=alpha,
          gamma=gamma,
          lambd=lambd,
          train_episodes=10000,
          eval_episodes=1000
      )

      self.results.append((alpha, gamma, lambd, success_rate))
      print(f"Success rate: {success_rate:.2f}%")

  def summary(self):
    top_5 = sorted(self.results, key=lambda x: x[3], reverse=True)[:5]
    print("Top 5 results:")
    for alpha, gamma, lambd, success_rate in top_5:
      print(f"alpha={alpha}, gamma={gamma}, lambd={lambd}, success_rate={success_rate:.2f}%")

  def train_and_evaluate(self, alpha, gamma, lambd, train_episodes, eval_episodes):
    model = TrueOnlineTDLambdaReplay(
        alpha=alpha,
        gamma=gamma,
        lambd=lambd,
        theta_init=np.zeros(self.state_space_size * self.action_space_size),
        n_obs=self.state_space_size,
        n_actions=self.action_space_size,
        action_space=self.env.action_space,
        feature_func=None
    )

    # Train the agent
    for episode in range(train_episodes):
      model.reset_traces()
      state, _ = self.env.reset()
      done = False

      action = env.action_space.sample()
      while not done:
        next_state, reward, terminal, truncated, info = self.env.step(action)
        done = terminal or truncated
        next_action = model.predict(state, action)
        model.update(state, action, reward, next_state, next_action, done)
        state = next_state
        action = next_action

    # Evaluate the agent
    success = []
    for _ in range(eval_episodes):
      state, _ = self.env.reset()
      done = False

      action = env.action_space.sample()
      while not done:
        action = model.predict(state, action)
        next_state, reward, terminal, truncated, info = self.env.step(action)
        done = terminal or truncated
        state = next_state

      success.append(1 if info.get("success", False) else 0)

    success_rate = np.sum(success) / eval_episodes * 100
    return success_rate


if __name__ == "__main__":
  # Test TrueOnlineTDLambdaReplay
  from Environments.FrozenLake import make_frozen_lake

  env = make_frozen_lake()

  cv = TrueOnlineTDLambdaReplayCV(env, param_grid=dict(
      alpha=[0.01, 0.05, 0.1, 0.5],
      gamma=[0.9, 0.95, 0.99],
      lambd=[0.5, 0.7, 0.9]
  ))
  cv.search()
  cv.summary()
