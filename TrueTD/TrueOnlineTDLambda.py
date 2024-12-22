import numpy as np


class TrueOnlineTDLambda:
  def __init__(self, num_features, alpha, gamma, lambd):
    self.num_features = num_features
    self.alpha = alpha  # Step size
    self.gamma = gamma  # Discount factor
    self.lambd = lambd  # Lambda for eligibility traces
    self.theta = np.zeros(num_features)  # Weight vector
    self.e = np.zeros(num_features)  # Eligibility trace
    self.v_old = 0  # Previous value estimate

  def update(self, phi, reward, phi_next, terminal):
    """
    Perform an update for a single time step.

    :param phi: Current feature vector.
    :param reward: Observed reward.
    :param phi_next: Next feature vector.
    :param terminal: Whether the next state is terminal.
    """
    v = np.dot(self.theta, phi)  # Current value estimate
    v_next = 0 if terminal else np.dot(self.theta, phi_next)  # Next value estimate

    delta = reward + self.gamma * v_next - v  # TD error

    # Update eligibility traces
    self.e = self.gamma * self.lambd * self.e + self.alpha * phi
    self.e -= self.alpha * self.gamma * self.lambd * (np.dot(self.e, phi)) * phi

    # Update weights
    self.theta += delta * self.e + self.alpha * (v - self.v_old) * phi
    self.v_old = v_next if not terminal else 0

  def predict(self, phi):
    """
    Get the estimated value for a state.

    :param phi: Feature vector of the state.
    :return: Estimated value.
    """
    return np.dot(self.theta, phi)

  def reset(self):
    """
    Reset the eligibility traces and old value for a new episode.
    """
    self.e.fill(0)
    self.v_old = 0


class TrueOnlineTDLambdaReplay:
  def __init__(self, alpha, gamma, lambd, theta_init, n_obs, n_actions, action_space, feature_func):
    self.alpha = alpha
    self.gamma = gamma
    self.lambd = lambd
    self.theta = theta_init.copy()
    self.feature_func = feature_func
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
