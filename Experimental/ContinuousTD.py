import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from Specs import AlgoSpecs


class IdentityApproximator:
  def __init__(self, **kwargs):
    pass

  def predict(self, state):
    return [0.0, 0.0]

  def update(self, **kwargs):
    pass


class DeepNNApproximator(nn.Module):
  def __init__(self, state_dim=8, action_dim=2, hidden_layers=[64, 64], lr=0.001):
    super(DeepNNApproximator, self).__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim

    # Define the neural network architecture
    layers = []
    input_dim = state_dim
    for hidden_dim in hidden_layers:
      layers.append(nn.Linear(input_dim, hidden_dim))
      layers.append(nn.ReLU())
      input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, action_dim))  # Output layer
    self.model = nn.Sequential(*layers)

    # Optimizer
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, state):
    return self.model(state).clamp(-1.0, 1.0)  # Clip output to [-1, 1]

  def predict(self, state):
    """Predict the action values for a given state."""
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
      return self(state_tensor).squeeze(0).numpy()

  def update(self, state, action, reward, next_state, done, gamma=0.99, **kwargs):
    """
    Update the model using a TD error-based approach.
    """
    # Convert inputs to tensors
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

    # Forward pass for current and next states
    prediction = self(state_tensor)
    next_prediction = self(next_state_tensor).detach()  # Detach to avoid backprop through next state

    # Compute TD target
    td_target = reward + (1 - done) * gamma * next_prediction
    td_target = td_target.squeeze(0)  # Remove batch dimension

    # Compute TD error
    td_error = td_target - prediction

    # Loss is the square of TD error
    loss = (td_error ** 2).mean()

    # Backward pass and optimizer step
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


class ContinuousMLPApproximator(nn.Module):
  def __init__(self, input_dim=8, output_dim=2, hidden_layers=None, lr=0.001):
    super(ContinuousMLPApproximator, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim

    # Default architecture if none provided
    if hidden_layers is None:
      hidden_layers = [64, 64]

    # Define fully connected layers
    layers = []
    in_features = input_dim
    for units in hidden_layers:
      layers.append(nn.Linear(in_features, units))
      layers.append(nn.ReLU())
      in_features = units
    layers.append(nn.Linear(in_features, output_dim))

    self.fc_layers = nn.Sequential(*layers)

    # Optimizer
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, state):
    return self.fc_layers(state).clamp(-1.0, 1.0)  # Clip actions to [-1, 1]

  def predict(self, state):
    """Predict the action values for a given state."""
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
      return self(state_tensor).squeeze(0).numpy()

  def update(self, state, action, reward, next_state, done, gamma=0.99, **kwargs):
    """
    Update the model using a TD error-based approach.
    """
    # Convert inputs to tensors
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

    # Forward pass for current and next states
    prediction = self(state_tensor)
    next_prediction = self(next_state_tensor).detach()  # Detach to avoid backprop through next state

    # Compute TD target
    td_target = reward + (1 - done) * gamma * next_prediction
    td_target = td_target.squeeze(0)  # Remove batch dimension

    # Compute TD error
    td_error = td_target - prediction

    # Loss is the square of TD error
    loss = (td_error ** 2).mean()

    # Backward pass and optimizer step
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


class ProbabilisticNNApproximator(nn.Module):
  def __init__(self, state_dim=8, action_dim=2, hidden_layers=[64, 64], lr=0.001):
    super(ProbabilisticNNApproximator, self).__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim

    # Define the neural network architecture
    layers = []
    input_dim = state_dim
    for hidden_dim in hidden_layers:
      layers.append(nn.Linear(input_dim, hidden_dim))
      layers.append(nn.ReLU())
      input_dim = hidden_dim

    self.hidden = nn.Sequential(*layers)

    # Output layers for mean and log-variance
    self.mean_layer = nn.Linear(input_dim, action_dim)
    self.log_var_layer = nn.Linear(input_dim, action_dim)

    # Optimizer
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, state):
    x = self.hidden(state)
    mean = self.mean_layer(x)
    log_var = self.log_var_layer(x)
    return mean, log_var

  def predict(self, state, deterministic=True):
    """Predict action values as mean or samples from the predicted distribution."""
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
      mean, log_var = self(state_tensor)
      std = torch.exp(0.5 * log_var)
      if deterministic:
        return mean.squeeze(0).numpy()
      else:
        dist = torch.distributions.Normal(mean, std)
        return dist.sample().squeeze(0).numpy()

  def update(self, state, action, reward, next_state, done, gamma=0.99, **kwargs):
    """Update the model using temporal difference (TD) methods."""
    # Convert inputs to tensors
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

    # Forward pass for current and next states
    mean, log_var = self(state_tensor)
    next_mean, next_log_var = self(next_state_tensor)
    std = torch.exp(0.5 * log_var)

    # Compute TD target
    next_value = next_mean.detach()  # Do not backpropagate through next value
    td_target = reward + (1 - done) * gamma * next_value

    # Negative log-likelihood loss
    dist = torch.distributions.Normal(mean, std)
    loss = -dist.log_prob(td_target).mean()

    # Backward pass and optimizer step
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


class NNApproximator(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001):
    super(NNApproximator, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim)
    )
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, state):
    return self.model(state)

  def predict(self, state):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
      return self(state_tensor).numpy()

  def update(self, **kwargs):
    state = kwargs['state']
    td_error = kwargs['td_error']

    state_tensor = torch.tensor(state, dtype=torch.float32)
    td_error_tensor = torch.tensor(td_error, dtype=torch.float32)

    prediction = self(state_tensor)
    loss = (td_error_tensor - prediction).pow(2).mean()

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


class FourierApproximator:
  def __init__(self, state_dim, action_dim, order=3, alpha=0.01):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.order = order
    self.alpha = alpha
    self.coefficients = np.random.uniform(-0.1, 0.1, ((order + 1) ** state_dim, action_dim))

  def _compute_fourier_features(self, state):
    terms = [np.cos(np.pi * np.dot(state, c)) for c in self._generate_coefficients()]
    return np.array(terms)

  def _generate_coefficients(self):
    return np.array(np.meshgrid(*[range(self.order + 1)] * self.state_dim)).T.reshape(-1, self.state_dim)

  def predict(self, state):
    features = self._compute_fourier_features(state)
    return np.dot(features, self.coefficients)

  def update(self, **kwargs):
    state = kwargs['state']
    td_error = kwargs['td_error']

    features = self._compute_fourier_features(state)
    gradient = np.outer(features, td_error)
    self.coefficients += self.alpha * gradient


class RNNApproximator(nn.Module):
  def __init__(self, input_dim=8, action_dim=2, hidden_dim=64, num_layers=1, lr=0.001):
    super(RNNApproximator, self).__init__()
    self.input_dim = input_dim
    self.action_dim = action_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers

    # RNN layers
    self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

    # Fully connected layer for output
    self.fc = nn.Linear(hidden_dim, action_dim)

    # Optimizer
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x, hidden=None):
    # Ensure hidden state matches the batch size of x
    batch_size = x.size(0)
    if hidden is None:
      hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

    # Forward pass through RNN
    out, hidden = self.rnn(x, hidden)
    # Only the last output step is passed to the fully connected layer
    out = self.fc(out[:, -1, :])
    # Clip output to action range [-1, 1]
    return out.clamp(-1.0, 1.0), hidden

  def predict(self, state_sequence, hidden=None):
    """Predict action values for a sequence of states."""
    state_tensor = torch.tensor(state_sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
      out, hidden = self(state_tensor, hidden)
      return out.squeeze(0).numpy(), hidden

  def update(self, state_sequence, action, reward, next_state_sequence, done, gamma=0.99):
    """
    Update the model using TD error.
    """
    # Convert inputs to tensors
    state_tensor = torch.tensor(state_sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    next_state_tensor = torch.tensor(next_state_sequence, dtype=torch.float32).unsqueeze(0)

    # Initialize hidden state
    hidden = torch.zeros(self.num_layers, 1, self.hidden_dim)

    # Forward pass for current and next states
    prediction, _ = self(state_tensor, hidden)
    next_prediction, _ = self(next_state_tensor, hidden)
    next_prediction = next_prediction.detach()  # Detach to avoid backprop through next state

    # Compute TD target
    td_target = reward + (1 - done) * gamma * next_prediction
    td_target = td_target.squeeze(0)  # Remove batch dimension

    # Compute TD error
    td_error = td_target - prediction

    # Loss is the square of TD error
    loss = (td_error ** 2).mean()

    # Backward pass and optimizer step
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()


class TApproximator:
  def __init__(self, state_dim, action_dim, weight_init_range=(-0.1, 0.1), alpha=0.01, **kwargs):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.T = np.random.uniform(weight_init_range[0], weight_init_range[1], (state_dim, action_dim))
    self.alpha = alpha

  def predict(self, state):
    return np.dot(state, self.T)

  def update(self, **kwargs):
    state = kwargs.get("state")
    action = kwargs.get("action")
    td_error = kwargs.get("td_error")

    gradient = np.outer(state, td_error)
    self.T += self.alpha * gradient
    self.T = np.clip(self.T, -1, 1)


class NewtonRaphsonApproximator:
  def __init__(self, state_dim, action_dim, alpha=0.01):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.weights = np.random.uniform(-0.1, 0.1, (state_dim, action_dim))  # Initialize weights
    self.alpha = alpha  # Step size for Newton-Raphson updates

  def predict(self, state):
    return np.dot(state, self.weights)

  def update(self, **kwargs):
    state = kwargs.get("state")
    action = kwargs.get("action")
    td_error = kwargs.get("td_error")

    # Compute the gradient (Jacobian matrix for the linear model)
    gradient = np.outer(state, td_error)

    # Compute the Hessian (second derivative approximation for quadratic loss)
    hessian = np.outer(state, state)

    # Invert Hessian with a regularization term for numerical stability
    hessian_inv = np.linalg.inv(hessian + 1e-6 * np.eye(hessian.shape[0]))

    # Newton-Raphson update step
    self.weights += self.alpha * hessian_inv @ gradient

    # Optional: Clip weights for numerical stability
    self.weights = np.clip(self.weights, -1, 1)


class LinearApproximator:
  def __init__(self, state_dim, action_dim, alpha=0.01):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.weights = np.random.uniform(-0.1, 0.1, (state_dim, action_dim))
    self.alpha = alpha

  def predict(self, state):
    return np.dot(state, self.weights)

  def update(self, **kwargs):
    state = kwargs['state']
    td_error = kwargs['td_error']

    gradient = np.outer(state, td_error)
    self.weights += self.alpha * gradient


class PolynomialApproximator:
  def __init__(self, state_dim, action_dim, degree=2, alpha=0.01):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.degree = degree
    self.alpha = alpha
    self.weights = np.random.uniform(-0.1, 0.1, ((degree + 1) ** state_dim, action_dim))

  def _compute_polynomial_features(self, state):
    return np.polynomial.polynomial.polyvander(state, self.degree).flatten()

  def predict(self, state):
    features = self._compute_polynomial_features(state)
    return np.dot(features, self.weights)

  def update(self, **kwargs):
    state = kwargs['state']
    td_error = kwargs['td_error']

    features = self._compute_polynomial_features(state)
    gradient = np.outer(features, td_error)
    self.weights += self.alpha * gradient


class GaussianApproximator:
  def __init__(self, state_dim, action_dim, num_kernels=10, alpha=0.01, sigma=1.0):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.num_kernels = num_kernels
    self.alpha = alpha  # Learning rate
    self.sigma = sigma  # Spread of the Gaussian kernels

    # Randomly initialize kernel centers and weights
    self.kernel_centers = np.random.uniform(-1, 1, (num_kernels, state_dim))
    self.weights = np.random.uniform(-0.1, 0.1, (num_kernels, action_dim))

  def _gaussian(self, state, center):
    return np.exp(-np.linalg.norm(state - center) ** 2 / (2 * self.sigma ** 2))

  def _compute_features(self, state):
    return np.array([self._gaussian(state, center) for center in self.kernel_centers])

  def predict(self, state):
    features = self._compute_features(state)
    return np.dot(features, self.weights)

  def update(self, **kwargs):
    state = kwargs['state']
    td_error = kwargs['td_error']

    # Compute features (Gaussian kernel activations)
    features = self._compute_features(state)

    # Gradient update
    gradient = np.outer(features, td_error)
    self.weights += self.alpha * gradient


class RBFApproximator:
  def __init__(self, state_dim, action_dim, num_centers=10, gamma=1.0):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.num_centers = num_centers
    self.gamma = gamma
    self.centers = np.random.uniform(-1, 1, (num_centers, state_dim))
    self.weights = np.random.uniform(-0.1, 0.1, (num_centers, action_dim))

  def _gaussian_rbf(self, state, center):
    return np.exp(-self.gamma * np.linalg.norm(state - center) ** 2)

  def _compute_features(self, state):
    return np.array([self._gaussian_rbf(state, center) for center in self.centers])

  def predict(self, state):
    features = self._compute_features(state)
    return np.dot(features, self.weights)

  def update(self, **kwargs):
    state = kwargs['state']
    td_error = kwargs['td_error']

    features = self._compute_features(state)
    gradient = np.outer(features, td_error)
    self.weights += kwargs['alpha'] * gradient


class PureRBFApproximator:
  def __init__(self, state_dim, action_dim, num_rbf=10, alpha=0.01, sigma=1.0):
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.num_rbf = num_rbf
    self.alpha = alpha  # Learning rate
    self.sigma = sigma  # Spread of the RBF kernels

    # Initialize RBF centers and weights
    self.centers = np.random.uniform(-1, 1, (num_rbf, state_dim))
    self.weights = np.random.uniform(-0.1, 0.1, (num_rbf, action_dim))

  def _rbf(self, state, center):
    """Compute the RBF value for a given state and center."""
    return np.exp(-np.linalg.norm(state - center) ** 2 / (2 * self.sigma ** 2))

  def _compute_features(self, state):
    """Compute RBF features for the given state."""
    return np.array([self._rbf(state, center) for center in self.centers])

  def predict(self, state):
    """Predict action values based on RBF features and weights."""
    features = self._compute_features(state)
    return np.dot(features, self.weights)

  def update(self, **kwargs):
    """Update the weights based on the TD error and features."""
    state = kwargs['state']
    td_error = kwargs['td_error']

    # Compute features (RBF activations)
    features = self._compute_features(state)

    # Gradient update rule
    gradient = np.outer(features, td_error)
    self.weights += self.alpha * gradient


class ContinuousTD(AlgoSpecs):
  def __init__(self, observation_space, action_space, approximator, alpha=0.01, epsilon=1.0, decay_rate=0.99, gamma=0.99, log=False, **kwargs):
    self.observation_space = observation_space
    self.action_space = action_space
    self.approximator = approximator
    self.alpha = alpha
    self.epsilon = epsilon
    self.gamma = gamma
    self.decay_rate = decay_rate
    self.log = log
    self.state_space_size = observation_space.shape[0]
    self.action_space_size = action_space.shape[0]
    self.td_errors = []

  def predict(self, state):
    pred = self.approximator.predict(state)
    bounded = np.clip(pred, self.action_space.low, self.action_space.high)
    return bounded

  def policy(self, state):
    if np.random.rand() < self.epsilon:
      pred = np.random.uniform(self.action_space.low, self.action_space.high, self.action_space_size)
    else:
      pred = self.approximator.predict(state)

      # Action is two floats [main engine, left-right engines].
      # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
      # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
      pred = np.clip(pred, self.action_space.low, self.action_space.high)

    return pred

  def update(self, **kwargs):
    state = kwargs.get("state")
    action = kwargs.get("action")
    reward = kwargs.get("reward")
    next_state = kwargs.get("next_state")
    next_action = kwargs.get("next_action")
    done = kwargs.get("done")

    self._update(state, action, reward, next_state, next_action, done)

  def _update(self, state, action, reward, next_state, next_action, done):
    action_value = self.approximator.predict(state)
    if done:
      target_action = reward
    else:
      next_value = self.approximator.predict(next_state)
      target_action = reward + self.gamma * next_value

    td_error = target_action - action_value

    self.td_errors.append(td_error)

    self.approximator.update(
        state=state,
        action=action,
        td_error=td_error,
        target_action=target_action,
        reward=reward,
        next_state=next_state,
        done=done,
        gamma=self.gamma,
        alpha=self.alpha)

  def decay_epsilon(self):
    self.epsilon = max(self.epsilon * self.decay_rate, 0.01)


from CrossValidation import LunarLanderContinuousGridSearchCV


class ESCCV(LunarLanderContinuousGridSearchCV):
  def new(self, **params):
    approximator = params.get("approximator", IdentityApproximator())
    del params["approximator"]

    return ContinuousTD(
        observation_space=self.env.observation_space,
        action_space=self.env.action_space,
        approximator=approximator,
        **params
    )


if __name__ == "__main__":
  from Environments.RandomWalk import estimate_goal_probability
  from Environments.LunarLander import make_lunar_lander

  env = make_lunar_lander(continuous=True)

  approximators = [
      NNApproximator(env.observation_space.shape[0], env.action_space.shape[0]),
  ]

  for approximator in approximators:
    param_grid = {
        "alpha": [0.1, 0.01, 0.001],
        "epsilon": [1.0],
        "decay_rate": [0.995, 0.99],
        "gamma": [0.99, 0.5],
        "approximator": [approximator]
    }

    cv = ESCCV(env, param_grid)
    cv.search(episodes=10000)

    # estimate_goal_probability(env)
    cv.summary()
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = f"results/{approximator.__class__.__name__}{now}.png"
    cv.plot_metrics(save_path)
    print(f"Saved plot to {save_path}")
