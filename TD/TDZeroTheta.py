
import numpy as np

from CrossValidation import GridSearchCV

from Specs import AlgoSpecs


class TDZeroTheta(AlgoSpecs):
  def __init__(self, action_space, observation_space, nrow, ncol, alpha=0.1, gamma=0.99, epsilon=1.0, decay_rate=0.99, policy=None, **kwargs):
    super().__init__(**kwargs)
    self.action_space = action_space
    self.observation_space = observation_space
    self.nrow = nrow
    self.ncol = ncol
    self.alpha = alpha
    self.gamma = gamma
    self.epsilon = epsilon
    self.decay_rate = decay_rate

    self.state_space_size = nrow * ncol
    self.num_actions = action_space.n  # Number of discrete actions
    self.feature_dim = feature_dim = 2 + self.num_actions  # Adjusted based on the feature vector logic

    # Initialize theta as a weight vector for parameterized approximation
    self.theta = np.random.rand(feature_dim)

    # Define the policy
    self.policy = self.egreedy_policy if policy is None else policy

  def _get_valid_actions(self, state):
    """Return a list of valid actions within the environment's bounds."""
    adjacent_indices = [
        state - 1,              # Left
        state + self.ncol,      # Down
        state + 1,              # Right
        state - self.ncol       # Up
    ]

    valid_actions = []
    for action, idx in enumerate(adjacent_indices):
      # Check if the action leads to a valid state index
      if 0 <= idx < self.state_space_size:
        # Check grid constraints for edges and corners
        if (action == 0 and state % self.ncol != 0) or \
           (action == 2 and state % self.ncol != self.ncol - 1) or \
           (action in [1, 3]):  # Up and Down have no column restrictions
          valid_actions.append(action)
    return valid_actions

  def feature_vector(self, state, action):
    """Return a feature vector for the given state-action pair."""
    state_features = np.array([state / self.state_space_size, 1.0])  # Normalize state
    action_features = np.eye(self.num_actions)[action]  # One-hot encode action
    return np.concatenate([state_features, action_features])[:self.feature_dim]

  def predict_q(self, state, action):
    """Predict Q(s, a) using the parameterized theta."""
    phi = self.feature_vector(state, action)
    return np.dot(self.theta, phi)

  def predict(self, state):
    """Predict the best action for a given state."""
    q_values = [self.predict_q(state, a) for a in self._get_valid_actions(state)]
    action = np.argmax(q_values)
    return action

  def egreedy_policy(self, state):
    """Epsilon-greedy policy for action selection."""
    if np.random.rand() < self.epsilon:
      return self.action_space.sample()  # Random valid action (exploration)
    else:
      # Exploit: choose action with the highest Q-value
      q_values = [self.predict_q(state, a) for a in self._get_valid_actions(state)]
      return np.argmax(q_values)

  def decay_epsilon(self):
    """Decay epsilon for exploration/exploitation balance."""
    self.epsilon = max(self.epsilon * self.decay_rate, 0.01)

  def reset(self):
    """Reset the agent state and decay epsilon."""
    self.decay_epsilon()

  def update(self, state, action, reward, next_state, done, **ignored):
    """Update theta using the TD(0) update rule."""
    # Compute current Q-value
    phi = self.feature_vector(state, action)
    current_q = np.dot(self.theta, phi)

    # Compute next state's maximum Q-value
    if done:
      next_q = 0
    else:
      next_q = max(self.predict_q(next_state, a) for a in range(self.num_actions))

    # TD error
    td_error = reward + self.gamma * next_q - current_q
    self.td_errors.append(td_error)

    # Update theta using gradient descent
    self.theta += self.alpha * td_error * phi


class SearchCV(GridSearchCV):
  def new(self, **params):
    return TDZeroTheta(
        action_space=self.env.action_space,
        observation_space=self.env.observation_space,
        nrow=self.env.nrow,
        ncol=self.env.ncol,
        **params
    )


if __name__ == "__main__":
  from Environments.RandomWalk import make_random_walk, estimate_goal_probability
  from Environments.FrozenLake import make_frozen_lake

  env = make_random_walk()  # Best results: alpha=0.003, gamma=0.7
  env = make_frozen_lake()  # Best results: alpha=0.003, gamma=0.2

  param_grid = {
      "alpha": np.linspace(0.0001, 1.0, 50),
      "gamma": np.linspace(0.0001, 1.0, 50),
  }

  cv = SearchCV(env, param_grid)
  cv.search()

  estimate_goal_probability(env)
  cv.summary()
