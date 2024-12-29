import numpy as np
import gymnasium as gym
from datetime import datetime

from Specs import AlgoSpecs

from CrossValidation import ContinuousGridSearchCV, withSave


class ActorCriticRBF(AlgoSpecs):
  def __init__(
      self,
      observation_space,
      action_space,
      alpha=0.01,
      gamma=0.99,
      tau=1.0,
      sigma_0=0.01,
      epsilon=0.01,
      decay_rate=0.01,
      grid_size=12,
      rbf_variance=0.1,
      **kwargs
  ):
    """
    Initializes the Actor-Critic model with Radial Basis Functions.

    Parameters:
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
        alpha (float): Base learning rate for the actor.
        gamma (float): Discount factor for critic updates.
        tau (float): Time constant for prediction updates.
        sigma_0 (float): Initial exploration noise scale.
        epsilon (float): Exploration noise (Gaussian).
        decay_rate (float): Decay rate for exploration noise.
        grid_size (int): Number of RBFs per dimension.
        rbf_variance (float): Variance of the RBFs.
    """
    self.observation_space = observation_space
    self.action_space = action_space
    self.alpha = alpha  # Actor's learning rate
    self.alpha_critic = alpha * 10  # Critic's learning rate
    self.gamma = gamma  # Discount factor
    self.tau = tau      # Time constant
    self.sigma = sigma_0  # Initial noise scale
    self.epsilon = epsilon  # Exploration noise
    self.decay_rate = decay_rate  # Noise decay rate

    # Define RBF grid for state space
    self.state_dim = observation_space.shape[0]
    self.action_dim = action_space.shape[0]

    # Generate RBF centers and variances
    grid_points = [np.linspace(low, high, grid_size) for low, high in zip(observation_space.low, observation_space.high)]
    self.centers = np.array(np.meshgrid(*grid_points)).T.reshape(-1, self.state_dim)
    self.num_rbf = self.centers.shape[0]

    self.variances = np.ones((self.num_rbf, self.state_dim)) * rbf_variance

    # Initialize weights for actor and critic
    self.critic_weights = np.random.uniform(-0.1, 0.1, self.num_rbf)
    self.actor_weights = np.random.uniform(-0.1, 0.1, (self.num_rbf, self.action_dim))

    # Store TD errors
    self.td_errors = []

  def rbf(self, state):
    """Compute RBF activations for a given state."""
    diff = state - self.centers  # Broadcasting works correctly now
    return np.exp(-np.sum((diff ** 2) / (2 * self.variances), axis=1))

  def policy(self, state):
    """Generate action using actor policy with noise for exploration."""
    rbf_values = self.rbf(state)
    deterministic_action = np.dot(rbf_values, self.actor_weights)
    noise = self.sigma * np.random.normal(size=self.action_dim)
    action = np.tanh(deterministic_action + noise)
    return np.clip(action, self.action_space.low, self.action_space.high)

  def predict(self, state):
    """Predict action without noise for evaluation."""
    rbf_values = self.rbf(state)
    deterministic_action = np.dot(rbf_values, self.actor_weights)
    return np.clip(np.tanh(deterministic_action), self.action_space.low, self.action_space.high)

  def critic_value(self, state):
    """Predict the value function for a given state."""
    rbf_values = self.rbf(state)
    return np.dot(self.critic_weights, rbf_values)

  def update(self, state, action, reward, next_state, delta_t=0.1, **kwargs):
    """Perform actor and critic updates based on TD error."""
    # Compute RBF activations
    rbf_current = self.rbf(state)
    rbf_next = self.rbf(next_state)

    # Temporal Difference (TD) error
    predicted_value = np.dot(self.critic_weights, rbf_current)
    predicted_value_next = np.dot(self.critic_weights, rbf_next)
    td_error = reward + self.gamma * predicted_value_next - predicted_value

    # Update critic weights
    self.critic_weights += self.alpha_critic * td_error * rbf_current

    # Update actor weights
    noise = np.random.normal(size=self.action_dim)
    policy_gradient = np.outer(rbf_current, td_error * noise)
    self.actor_weights += self.alpha * policy_gradient

    # Decay noise scale
    self.sigma = max(self.sigma * (1 - self.decay_rate), self.epsilon)

    # Store TD error for analysis
    self.td_errors.append(td_error)

  def save(self, filename):
    """Save the model parameters to a file."""
    filename = f"{filename}.npz"
    data = {
        "centers": self.centers,
        "variances": self.variances,
        "critic_weights": self.critic_weights,
        "actor_weights": self.actor_weights
    }
    np.savez(filename, **data)
    return filename

  def load(self, path):
    """Load the model parameters from a file."""
    filename = f"{path}.npz"
    data = np.load(filename)
    self.centers = data["centers"]
    self.variances = data["variances"]
    self.critic_weights = data["critic_weights"]
    self.actor_weights = data["actor_weights"]


def train(eps=100):
  from Environments.Pendulum import make_pendulum
  env = make_pendulum()
  ac_rbf = ActorCriticRBF(env.observation_space, env.action_space)

  for ep in range(eps):
    state, _ = env.reset()
    done = False
    while not done:
      action = ac_rbf.policy(state)
      next_state, reward, done, truncated, info = env.step(action)
      done = done or truncated
      ac_rbf.update(state, action, reward, next_state)
      state = next_state
      success = info.get("success", False)
      if success:
        print("Success!")
    print(f"Episode {ep + 1}: {info}")


class Search(ContinuousGridSearchCV):
  def new(self, **params):
    return ActorCriticRBF(
        observation_space=self.env.observation_space,
        action_space=self.env.action_space,
        **params
    )


if __name__ == "__main__":
  now = datetime.now().strftime("%Y%m%d%H%M%S")

  from Environments.Pendulum import make_pendulum
  param_grid = {
      "alpha": [0.001, 0.005, 0.01],
      "epsilon": [0.001, 0.005, 0.01, 0.02, 0.05],
      "decay_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
      "gamma": [0.99],
      "tau": [0.5, 1.0, 2.0],
      "grid_size": [12, 16],
      "rbf_variance": [0.01, 0.05, 0.1, 0.2],
      "max_torque": [10]
  }

  env = make_pendulum()

  withSave(Search(env, param_grid), model_path="Experimental.Doya", episodes=4000, samples=3)
