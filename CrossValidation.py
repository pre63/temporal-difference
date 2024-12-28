
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from multiprocessing import Pool, get_context
from itertools import product

from Specs import SearchCV


import os
from datetime import datetime

from Replay import save_config


def withSave(cv, save_dir="models", model_path=None, **search_params):
  """
  Executes a complete workflow: search, evaluation, summary, saving plots, saving the model, 
  and saving the configuration.

  Parameters:
      cv: The cross-validation object with the necessary methods (search, summary, plot_metrics, get_best_model).
      save_dir (str): The base directory for saving models and configurations.
      **search_params (dict): Optional parameters to override the default search behavior.
  """
  # Perform the search with the provided parameters
  if search_params is None:
    search_params = {"episodes": 1000}
  cv.search(**search_params)

  # Generate a summary
  cv.summary()

  # Create a timestamped directory for saving results
  now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  path = os.path.join(save_dir, f"Experiment-{now}")
  os.makedirs(path, exist_ok=True)

  # Save plots
  save_path = os.path.join(path, "metrics.png")
  save_path_best = os.path.join(path, "metrics_best.png")

  cv.plot_metrics(save_path=save_path)
  print(f"Saved plot to {save_path}")

  cv.plot_metrics(save_path=save_path_best, best_model=True)
  print(f"Saved plot to {save_path_best}")

  # Get the best model
  model = cv.get_best_model()
  print(f"Best Model: {model}, {type(model)}, {model.__class__.__name__}, module: {model.__module__}")

  # Derive configuration details
  model_path = model_path if not None else model.__module__  # Dynamically get the module path of the model
  model_class = model.__class__.__name__  # Dynamically get the model class name
  env_module = cv.env.__module__  # Dynamically get the module path of the environment
  env_name = cv.env.make_func_name  # Dynamically get the environment creation function name

  # Save the configuration
  model_file_path = save_config(path, model_path, model_class, env_module, env_name)

  # Save the model to the derived path
  model.save(model_file_path)

  # Print replay command
  print("Replay:\n")
  print(f"    python Replay.py {path}\n")


class GridSearchCV(SearchCV):
  def __init__(self, env, param_grid):
    self.env = env
    self.param_grid = param_grid
    self.results = []
    self.episode_rewards = []
    self.episode_steps = []
    self.value_function = []
    self.td_errors = []

    num_permution_for_all_params = np.prod([len(param_grid[key]) for key in param_grid.keys()])
    print("Number of permutations:", num_permution_for_all_params)
    print(f"Parameters")
    for key, values in param_grid.items():
      processed_values = [
          v if isinstance(v, (int, str, float)) else v.__class__.__name__
          for v in values
      ]
      print(f"  {key}: {processed_values}")

  def new(self, **params):
    pass

  def search(self, episodes=5000, samples=5):
    param_combinations = [
        dict(zip(self.param_grid.keys(), values))
        for values in product(*self.param_grid.values())
    ]

    # Use 'spawn' to avoid print suppression
    with get_context("spawn").Pool() as pool:
      self.results = pool.map(
          self._evaluate_params,
          [(params, episodes, samples) for params in param_combinations]
      )

  def _evaluate_model(self, model, trials=100):
    trials = max(trials, 1)
    success, rewards = 0, 0

    for _ in range(trials):
      state, _ = self.env.reset()
      done = False

      while not done:
        action = model.predict(state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        state = next_state

      rewards += reward
      success += 1 if info.get("success", False) else 0

    avg_success = (success / trials) * 100
    avg_rewards = rewards / trials

    return avg_success, avg_rewards

  def _evaluate_params(self, args):
    params, episodes, samples = args
    all_data = []

    # Train multiple models with the given parameters
    for _ in range(samples):
      model = self.new(**params)
      sample_data = self._train_and_collect_data(model, episodes)
      sample_data['model'] = model
      all_data.append(sample_data)

    # Select the best model based on the highest sum of successes
    best_sample = max(all_data, key=lambda x: sum(x['rewards']))
    best_model = best_sample['model']
    avg_success, avg_rewards = self._evaluate_model(best_model, episodes // 20)

    result = {
        'params': params,
        'avg_success': avg_success,
        'avg_rewards': avg_rewards,
        'avg_td_error': np.mean(best_model.td_errors) if hasattr(best_model, 'td_errors') else 0.0,
        'td_errors': best_model.td_errors if hasattr(best_model, 'td_errors') else [],
        'samples': all_data,
        'model': best_model
    }  # Compute the RMSE for TD errors

    avg_td_error = np.sqrt(np.mean(np.square(result['td_errors'])))

    formatted_params = {
        k: (v if isinstance(v, (int, str, float)) else v.__class__.__name__)
        for k, v in params.items()
    }

    print(f"Params: {formatted_params}, Avg Success: {avg_success:.2f}%, Avg Rewards: {avg_rewards:.2f}, RMSE TD Error: {avg_td_error:.2f}")

    return result

  def _train_and_collect_data(self, model, episodes):
    rewards, steps, success = [], [], []

    for _ in range(episodes):
      episode_rewards = []
      done = False

      model.reset()

      state, _ = self.env.reset()
      action = model.policy(state)

      while not done:
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_action = model.policy(next_state)
        done = terminated or truncated

        kwags = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'next_action': next_action,
            'done': done
        }

        model.update(**kwags)

        episode_rewards.append(reward)
        state = next_state
        action = next_action

      rewards.append(sum(episode_rewards))
      steps.append(len(episode_rewards))
      success.append(0 if info.get("success", False) else 1)

    value_function = self._generate_value_function(model)

    return {'rewards': rewards, 'steps': steps, 'success': sum(success), 'value_function': value_function}

  def _generate_value_function(self, model):
    """Generate the value function for visualization using the model's prediction."""
    value_func = []
    for state in range(self.env.observation_space.start, self.env.observation_space.n):
      value = model.predict(state)
      if self.env.observation_space.contains(value):
        value_func.append(value)
      else:
        value_func.append(None)  # Handle out-of-bounds predictions
    return value_func

  def get_best_model(self, search_function=None):
    if search_function is None:
      def search_function(results):
        result = sorted(results, key=lambda x: x['avg_success'], reverse=True)[0]
        sample = sorted(result['samples'], key=lambda x: x['success'], reverse=True)[0]
        return sample['model']

    return search_function(self.results)

  def summary(self, sort_lambda=None):
    """
    Print the top 5 results based on the average success rate.
    This funciton also sorts the results based on the provided lambda function.
    Parameters:
        sort_lambda (function): A lambda function to sort the results, defaults to sorting on average success rate.
    """
    if sort_lambda is None:
      def sort_lambda(x): return x['avg_success']

    self.results = sorted(self.results, key=sort_lambda, reverse=True)

    top_5 = self.results[:5]
    print("Top 5 Results:")
    for i, result in enumerate(top_5):
      formatted_params = {
          k: (v if isinstance(v, (int, str, float)) else v.__class__.__name__)
          for k, v in result['params'].items()
      }
      avg_td_error = np.sqrt(np.mean(np.square(result['td_errors']))) if 'td_errors' in result else float('nan')
      print(f"Run {i + 1}: Params: {formatted_params}, Avg Success: {result['avg_success']:.2f}%, Avg Rewards: {result['avg_rewards']:.2f}, RMSE TD Error: {avg_td_error:.2f}")

  def plot_metrics(self, save_path=None, index=None, best_model=False):
    """
    Plot the rewards, steps, value function, and TD errors for the top 5 results.
    Parameters:
        save_path (str): The path to save the plot as an image file.
        index (int): The index of the result to plot, defaults to the best result.
    """
    if index is not None:
      if index < 0 or index >= len(self.results):
        print(f"Invalid index: {index}. Please provide a valid index.")
        return
      result = self.results[index]
    else:
      top_5 = sorted(self.results, key=lambda x: x['avg_success'], reverse=True)[:5]
      if len(top_5) == 0:
        print("No results to plot")
        return
      result = top_5[0]

    samples = (
        [max(result['samples'], key=lambda s: s['success'])]
        if best_model else
        result['samples']
    )

    plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2)

    # Plot rewards per episode for all samples
    ax1 = plt.subplot(gs[0, 0])
    for j, sample in enumerate(samples):
      ax1.plot(sample['rewards'], label=f"Sample {j + 1}")
    ax1.set_title("Rewards per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Rewards")
    ax1.legend()

    # Plot steps per episode for all samples
    ax2 = plt.subplot(gs[0, 1])
    for j, sample in enumerate(samples):
      ax2.plot(sample['steps'], label=f"Sample {j + 1}")
    ax2.set_title("Steps per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.legend()

    # Plot value function for all samples
    ax3 = plt.subplot(gs[1, 0])
    for j, sample in enumerate(samples):
      ax3.plot(sample['value_function'], label=f"Sample {j + 1}")
    ax3.set_title("Value Function")
    ax3.set_xlabel("State Index")
    ax3.set_ylabel("Predicted Value")
    ax3.legend()

    # Plot TD Error for all samples and compute RMSE
    ax4 = plt.subplot(gs[1, 1])
    rmse_td_errors = []
    for j, sample in enumerate(samples):
      model = sample['model']  # Access the model object from each sample
      if hasattr(model, 'td_errors') and model.td_errors:
        td_errors = model.td_errors
        ax4.plot(td_errors, label=f"Sample {j + 1}")
        rmse = np.sqrt(np.mean(np.square(td_errors)))
        rmse_td_errors.append(rmse)

    # Compute average RMSE across samples
    avg_rmse_td_error = np.mean(rmse_td_errors) if rmse_td_errors else float('nan')
    ax4.set_title(f"TD Error (Avg RMSE: {avg_rmse_td_error:.2f})")
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("TD Error")
    ax4.legend()

    plt.tight_layout()
    if save_path is not None:
      plt.savefig(save_path)
    else:
      plt.show()


class ContinuousGridSearchCV(GridSearchCV):

  def _generate_value_function(self, model):
    """
    Generate the value function for continuous state-action spaces.

    Args:
        model: The model to use for value function prediction.

    Returns:
        np.ndarray: The value function approximations over sampled states.
    """
    sampled_states = np.linspace(
        self.env.observation_space.low, self.env.observation_space.high, num=100
    )
    value_func = []
    for state in sampled_states:
      state = np.array(state)
      value = model.predict(state)
      value_func.append(value)
    return value_func


class LunarLanderContinuousGridSearchCV(GridSearchCV):
  def _generate_value_function(self, model, resolution=100):
    """
    Generate the value function for continuous state-action spaces using a sigmoid gradient.

    Args:
        model: The model to use for value function prediction.
        resolution: Number of points to sample along the sigmoid gradient.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Grid of states and corresponding predictions.
    """
    # Define bounds for the first six state dimensions
    state_bounds = np.array([self.env.observation_space.low, self.env.observation_space.high]).T[:6]

    # Generate a sigmoid gradient from 0 to 1
    sigmoid = 1 / (1 + np.exp(-np.linspace(-6, 6, resolution)))  # Sigmoid over [-6, 6]

    # Scale the sigmoid gradient to match the bounds of each dimension
    gradient = [
        bounds[0] + sigmoid * (bounds[1] - bounds[0])
        for bounds in state_bounds
    ]

    # Generate all combinations where all six values increment simultaneously
    state_grid = np.array([gradient[dim] for dim in range(6)]).T

    # Fix other dimensions (dimensions 6 and 7) to representative values
    fixed_values = [0, 0]  # No leg contact
    fixed_states = np.hstack([
        state_grid,
        np.tile(fixed_values, (state_grid.shape[0], 1))
    ])

    # Predict values for each state in the grid
    predictions = []
    for state in fixed_states:
      value = model.predict(state)
      predictions.append(value)

    return np.array(predictions)


class LocalAdaptiveGridSearchCV(SearchCV):
  def __init__(self, env, param_grid, samples=5):
    self.env = env
    self.param_grid = param_grid
    self.results = {}
    self.visited = set()
    self.samples = samples

    # Convert the param_grid into a structured grid
    self.alpha_values = param_grid['alpha']
    self.gamma_values = param_grid['gamma']

  def new(self, **params):
    pass

  def search(self, episodes=4000, max_iterations=50, start_cell=None):
    if start_cell is None:
      start_cell = (
          np.random.randint(len(self.alpha_values)),
          np.random.randint(len(self.gamma_values))
      )

    current_cell = start_cell
    iteration = 0

    while iteration < max_iterations:
      iteration += 1

      alpha = self.alpha_values[current_cell[0]]
      gamma = self.gamma_values[current_cell[1]]
      params = {'alpha': alpha, 'gamma': gamma}

      print(f"Iteration {iteration}, Evaluating cell: Alpha={alpha:.4f}, Gamma={gamma:.4f}")

      if current_cell not in self.visited:
        self.visited.add(current_cell)
        fitness = self._evaluate_params(params, episodes)
        self.results[current_cell] = fitness

        if fitness == 100.0:
          print("Perfect score achieved! Terminating search.")
          break
      else:
        fitness = self.results[current_cell]

      print(f"Cell Performance: {fitness:.2f}")

      neighbors = self._get_neighbors(current_cell)
      best_neighbor = None
      best_fitness = -np.inf

      for neighbor in neighbors:
        if neighbor not in self.visited:
          alpha = self.alpha_values[neighbor[0]]
          gamma = self.gamma_values[neighbor[1]]
          neighbor_params = {'alpha': alpha, 'gamma': gamma}
          neighbor_fitness = self._evaluate_params(neighbor_params, episodes)

          self.visited.add(neighbor)
          self.results[neighbor] = neighbor_fitness

          if neighbor_fitness > best_fitness:
            best_fitness = neighbor_fitness
            best_neighbor = neighbor

      if best_neighbor is None:
        print("No better neighbors found. Terminating search.")
        break

      current_cell = best_neighbor
      print(f"Moving to best neighbor: {current_cell}, Fitness: {best_fitness:.2f}")

  def _evaluate_params(self, params, episodes):
    fitness_values = []
    for _ in range(self.samples):
      model = self.new(**params)
      avg_success, _ = self._evaluate_model(model, episodes // 20)
      fitness_values.append(avg_success)
    fitness = np.mean(fitness_values)
    return fitness

  def _evaluate_model(self, model, trials=100):
    success, rewards = 0, 0

    for _ in range(trials):
      state, _ = self.env.reset()
      done = False

      while not done:
        action = model.predict(state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        state = next_state

      rewards += reward
      success += 1 if info.get("success", False) else 0

    avg_success = (success / trials) * 100
    avg_rewards = rewards / trials
    return avg_success, avg_rewards

  def _get_neighbors(self, cell):
    neighbors = []
    x, y = cell
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
      nx, ny = x + dx, y + dy
      if 0 <= nx < len(self.alpha_values) and 0 <= ny < len(self.gamma_values):
        neighbors.append((nx, ny))
    return neighbors

  def summary(self):
    sorted_results = sorted(self.results.items(), key=lambda x: x[1], reverse=True)
    print("Best Results:")
    for cell, fitness in sorted_results[:5]:
      alpha = self.alpha_values[cell[0]]
      gamma = self.gamma_values[cell[1]]
      print(f"Cell: Alpha={alpha:.4f}, Gamma={gamma:.4f}, Fitness={fitness:.2f}")


class FullAdaptiveGridSearchCV(SearchCV):
  def __init__(self, env, param_grid, samples=5):
    self.env = env
    self.param_grid = param_grid
    self.results = {}
    self.visited = set()
    self.samples = samples

    # Convert the param_grid into a structured grid
    self.alpha_values = param_grid['alpha']
    self.gamma_values = param_grid['gamma']
    self.total_cells = len(self.alpha_values) * len(self.gamma_values)

  def new(self, **params):
    pass

  def search(self, episodes=4000, max_iterations=100, start_cell=None):
    if start_cell is None:
      start_cell = (
          np.random.randint(len(self.alpha_values)),
          np.random.randint(len(self.gamma_values))
      )

    current_cell = start_cell
    iteration = 0

    while iteration < max_iterations and len(self.visited) < self.total_cells:
      iteration += 1

      alpha = self.alpha_values[current_cell[0]]
      gamma = self.gamma_values[current_cell[1]]
      params = {'alpha': alpha, 'gamma': gamma}

      print(f"Iteration {iteration}, Evaluating cell: Alpha={alpha:.4f}, Gamma={gamma:.4f}")

      if current_cell not in self.visited:
        self.visited.add(current_cell)
        fitness = self._evaluate_params(params, episodes)
        self.results[current_cell] = fitness

        if fitness == 100.0:
          print("Perfect score achieved! Terminating search.")
          break
      else:
        fitness = self.results[current_cell]

      print(f"Cell Performance: {fitness:.2f}")

      neighbors = self._get_neighbors(current_cell)
      best_neighbor = None
      best_fitness = -np.inf

      for neighbor in neighbors:
        if neighbor not in self.visited:
          alpha = self.alpha_values[neighbor[0]]
          gamma = self.gamma_values[neighbor[1]]
          neighbor_params = {'alpha': alpha, 'gamma': gamma}
          neighbor_fitness = self._evaluate_params(neighbor_params, episodes)

          self.visited.add(neighbor)
          self.results[neighbor] = neighbor_fitness

          if neighbor_fitness > best_fitness:
            best_fitness = neighbor_fitness
            best_neighbor = neighbor

      if best_neighbor is None:
        unvisited_cells = [
            (i, j)
            for i in range(len(self.alpha_values))
            for j in range(len(self.gamma_values))
            if (i, j) not in self.visited
        ]
        if unvisited_cells:
          best_neighbor = random.choice(unvisited_cells)
          print("No better neighbors found. Moving to a random unvisited cell.")
        else:
          print("All cells visited.")
          break

      current_cell = best_neighbor
      print(f"Moving to next cell: {current_cell}")

  def _evaluate_params(self, params, episodes):
    fitness_values = []
    for _ in range(self.samples):
      model = self.new(**params)
      avg_success, _ = self._evaluate_model(model, episodes // 20)
      fitness_values.append(avg_success)
    fitness = np.mean(fitness_values)
    return fitness

  def _evaluate_model(self, model, trials=100):
    success, rewards = 0, 0

    for _ in range(trials):
      state, _ = self.env.reset()
      done = False

      while not done:
        action = model.predict(state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        state = next_state

      rewards += reward
      success += 1 if info.get("success", False) else 0

    avg_success = (success / trials) * 100
    avg_rewards = rewards / trials
    return avg_success, avg_rewards

  def _get_neighbors(self, cell):
    neighbors = []
    x, y = cell
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
      nx, ny = x + dx, y + dy
      if 0 <= nx < len(self.alpha_values) and 0 <= ny < len(self.gamma_values):
        neighbors.append((nx, ny))
    return neighbors

  def summary(self):
    sorted_results = sorted(self.results.items(), key=lambda x: x[1], reverse=True)
    print("Best Results:")
    for cell, fitness in sorted_results[:5]:
      alpha = self.alpha_values[cell[0]]
      gamma = self.gamma_values[cell[1]]
      print(f"Cell: Alpha={alpha:.4f}, Gamma={gamma:.4f}, Fitness={fitness:.2f}")


class GeneticSearchCV(SearchCV):
  def __init__(self, env, param_grid, population_size=20, mutation_rate=0.1, crossover_rate=0.7, samples=5):
    self.env = env
    self.param_grid = param_grid
    self.population_size = population_size
    self.mutation_rate = mutation_rate
    self.crossover_rate = crossover_rate
    self.samples = samples
    self.results = []

  def new(self, **params):
    pass

  def search(self, generations=10, episodes=4000):
    population = self._initialize_population()
    for generation in range(generations):
      print(f"Generation {generation + 1}/{generations}")

      with get_context("spawn").Pool() as pool:
        fitness = pool.map(
            self._evaluate_individual,
            [(individual, episodes) for individual in population]
        )

      # Pair individuals with their fitness and sort by fitness
      population_with_fitness = list(zip(population, fitness))
      population_with_fitness.sort(key=lambda x: x[1], reverse=True)

      # Save the best-performing individual of this generation
      self.results.append(population_with_fitness[0])
      print(f"Best Individual: {population_with_fitness[0][0]}, Fitness: {population_with_fitness[0][1]:.2f}")

      # Create the next generation
      next_population = [ind for ind, fit in population_with_fitness[:self.population_size // 2]]  # Elitism

      # Apply crossover
      while len(next_population) < self.population_size:
        if random.random() < self.crossover_rate:
          parent1, parent2 = random.sample(next_population[:self.population_size // 2], 2)
          offspring = self._crossover(parent1, parent2)
          next_population.append(offspring)

      # Apply mutation
      next_population = [self._mutate(ind) for ind in next_population]

      population = next_population

  def _initialize_population(self):
    population = []
    for _ in range(self.population_size):
      individual = {
          key: random.choice(values)
          for key, values in self.param_grid.items()
      }
      population.append(individual)
    return population

  def _evaluate_individual(self, args):
    individual, episodes = args
    fitness_values = []
    for _ in range(self.samples):
      model = self.new(**individual)
      avg_success, _ = self._evaluate_model(model, episodes // 20)
      fitness_values.append(avg_success)
    return np.mean(fitness_values)

  def _evaluate_model(self, model, trials=100):
    success, rewards = 0, 0

    for _ in range(trials):
      state, _ = self.env.reset()
      done = False

      while not done:
        action = model.predict(state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        state = next_state

      rewards += reward
      success += 1 if info.get("success", False) else 0

    avg_success = (success / trials) * 100
    avg_rewards = rewards / trials
    return avg_success, avg_rewards

  def _crossover(self, parent1, parent2):
    child = {}
    for key in parent1.keys():
      child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
    return child

  def _mutate(self, individual):
    mutated = individual.copy()
    for key in mutated.keys():
      if random.random() < self.mutation_rate:
        mutated[key] = random.choice(self.param_grid[key])
    return mutated

  def summary(self):
    print("Top Results:")
    sorted_results = sorted(self.results, key=lambda x: x[1], reverse=True)
    for i, (params, fitness) in enumerate(sorted_results[:5]):
      print(f"Rank {i + 1}: Params={params}, Fitness={fitness:.2f}")
