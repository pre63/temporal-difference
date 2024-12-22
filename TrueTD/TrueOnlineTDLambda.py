import numpy as np
from TD.TDZero import TDZero, TDZeroCV


class TrueOnlineTDLambda(TDZero):
  def __init__(self, action_space, observation_space, nrow, ncol, alpha, gamma, lambd, **kwargs):
    super().__init__(action_space, observation_space, nrow, ncol, **kwargs)
    num_features = observation_space.n
    self.num_features = num_features
    self.alpha = alpha  # Step size
    self.gamma = gamma  # Discount factor
    self.lambd = lambd  # Lambda for eligibility traces
    self.theta = np.zeros(num_features)  # Weight vector
    self.e = np.zeros(num_features)  # Eligibility trace
    self.v_old = 0  # Previous value estimate

  def update(self, phi, action, reward, phi_next, next_action, terminal):
    """
    Perform an update for a single time step.

    :param phi: Current feature vector.
    :param reward: Observed reward.
    :param phi_next: Next feature vector.
    :param terminal: Whether the next state is terminal.
    """
    phi = phi / (np.linalg.norm(phi) + 1e-10)  # Normalize phi to prevent large values
    phi_next = phi_next / (np.linalg.norm(phi_next) + 1e-10)  # Normalize phi_next to prevent large values

    v = np.dot(self.theta, phi)  # Current value estimate
    v_next = 0 if terminal else np.dot(self.theta, phi_next)  # Next value estimate

    delta = reward + self.gamma * v_next - v  # TD error

    # Update eligibility traces
    self.e = self.gamma * self.lambd * self.e + self.alpha * phi
    self.e -= self.alpha * self.gamma * self.lambd * (np.dot(self.e, phi)) * phi

    # Update weights
    self.theta += delta * self.e + self.alpha * (v - self.v_old) * phi
    self.v_old = v_next if not terminal else 0

  def predict(self, state):
    """
    Predict the best action based on the current state using theta.

    Parameters:
    - state (int): The current state index.

    Returns:
    - best_action (int): The best action to take based on predicted values.
    """
    if self.action_space.n == 2:
      # For two actions (e.g., WEST and EAST)
      adjacent_indices = [state - 1, state + 1]
      adjacent_states = [
          self.theta[idx] if 0 <= idx < self.state_space_size else -np.inf
          for idx in adjacent_indices
      ]
      best_action = int(np.argmax(adjacent_states))
      return best_action

    elif self.action_space.n == 4:
      # For four actions (e.g., UP, DOWN, LEFT, RIGHT)
      adjacent_indices = [
          state - 1,              # Left
          state + self.ncol,      # Down
          state + 1,              # Right
          state - self.ncol       # Up
      ]

      adjacent_states = []
      for i, idx in enumerate(adjacent_indices):
        if 0 <= idx < self.state_space_size:
          # Validate movement within grid boundaries
          if (i == 0 and state % self.ncol != 0) or (i == 2 and state % self.ncol != self.ncol - 1) or (i in [1, 3]):
            adjacent_states.append(self.theta[idx])

      best_index = int(np.argmax(adjacent_states))
      action = [0, 1, 2, 3][best_index]
      return action

  def reset(self):
    """
    Reset the eligibility traces and old value for a new episode.
    """
    self.e.fill(0)
    self.v_old = 0


class TrueOnlineTDLambdaCV(TDZeroCV):
  def new(self, **params):
    return TrueOnlineTDLambda(
        action_space=self.env.action_space,
        observation_space=self.env.observation_space,
        nrow=self.env.nrow,
        ncol=self.env.ncol,
        **params
    )


if __name__ == "__main__":
  from Environments.RandomWalk import make_random_walk, estimate_goal_probability
  from Environments.FrozenLake import make_frozen_lake

  env = make_frozen_lake()
  env = make_random_walk()  # Best Alpha: 0.0100, Gamma: 0.9500, Lambda: 0.9000

  param_grid = {
      "alpha": [0.01, 0.05, 0.1, 0.5],
      "gamma": [0.9, 0.95, 0.99],
      "lambd": [0.5, 0.7, 0.9]
  }

  cv = TrueOnlineTDLambdaCV(env, param_grid)
  cv.search()

  estimate_goal_probability(env)
  cv.summary()
  cv.plot_metrics()
