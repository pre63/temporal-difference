import numpy as np
from TD.TDZeroTheta import TDZeroTheta
from CrossValidation import GridSearchCV


class TDZeroThetaReplay(TDZeroTheta):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # Initialize eligibility matrix based on feature dimensionality
    self.eligibility_matrix = np.identity(self.feature_dim)

  def reset(self):
    """Reset eligibility matrix and decay epsilon."""
    super().reset()
    self.eligibility_matrix = np.identity(self.feature_dim)

    def update(self, state, action, reward, next_state, done, **ignored):
      """Update theta using replay mechanism and TD(0)."""

      # Compute feature vector for the current state-action pair
      phi = self.feature_vector(state, action)
      current_q = np.dot(self.theta, phi)

      # Compute next state's maximum Q-value
      if done:
        next_q = 0
      else:
        next_q = max(
            np.dot(self.theta, self.feature_vector(next_state, a))
            for a in self._get_valid_actions(next_state)
        )

      # TD Error
      td_error = reward + self.gamma * next_q - current_q
      self.td_errors.append(td_error)

      # Construct feature vector for next state (aggregated over all valid actions)
      phi_next = np.zeros_like(phi)
      if not done:
        for a in self._get_valid_actions(next_state):
          phi_next += self.feature_vector(next_state, a)

      # Update eligibility matrix
      self.eligibility_matrix = self.gamma * self.eligibility_matrix + np.outer(phi, phi)

      # Perform replay update
      replay_update = td_error * np.dot(self.eligibility_matrix, phi)

      # Update theta using replay-adjusted gradient
      self.theta += self.alpha * replay_update


class SearchCV(GridSearchCV):
  def new(self, **params):
    return TDZeroThetaReplay(
        action_space=self.env.action_space,
        observation_space=self.env.observation_space,
        nrow=self.env.nrow,
        ncol=self.env.ncol,
        **params
    )


if __name__ == "__main__":
  from Environments.RandomWalk import make_random_walk, estimate_goal_probability
  from Environments.FrozenLake import make_frozen_lake

  env = make_random_walk()  # alpha=0.005, gamma=1.0, success_rate=40.10%
  env = make_frozen_lake()  # alpha=0.002, gamma=0.6,  success_rate=90.00%

  param_grid = {
      "alpha": np.linspace(0.0001, 1.0, 50),
      "gamma": np.linspace(0.0001, 1.0, 50),
  }

  cvr = SearchCV(env, param_grid)
  cvr.search()

  estimate_goal_probability(env)
  cvr.summary()
