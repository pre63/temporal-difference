
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from multiprocessing import Pool, get_context
from itertools import product

from Specs import SearchCV


class GridSearchCV(SearchCV):
  def __init__(self, env, param_grid):
    self.env = env
    self.param_grid = param_grid
    self.results = []
    self.episode_rewards = []
    self.episode_steps = []
    self.value_function = []

    print("Number of permutations:", len(param_grid["alpha"]) * len(param_grid["gamma"]))
    print("Alpha values:", [float(f"{x:.4f}") for x in param_grid['alpha']])
    print("Gamma values:", [float(f"{x:.4f}") for x in param_grid['gamma']])

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
        'samples': all_data
    }

    print(f"Alpha: {params['alpha']:.4f}, Gamma: {params['gamma']:.4f}, Avg Success: {avg_success:.2f}%, Avg Rewards: {avg_rewards:.2f}")

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

  def summary(self):
    top_5 = sorted(self.results, key=lambda x: x['avg_success'], reverse=True)[:5]
    print("Top 5 Results:")
    for i, result in enumerate(top_5):
      print(f"Run {i + 1}: {result['params']}, Avg Success: {result['avg_success']:.2f}%, Avg Rewards: {result['avg_rewards']:.2f}")

  def plot_metrics(self, index=None):
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

    samples = result['samples']

    plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2)

    # Plot rewards per episode for all samples
    ax1 = plt.subplot(gs[0, 0])
    for j, sample in enumerate(samples):
      ax1.plot(sample['rewards'], label=f"Sample {j + 1}")
    ax1.set_title(f"Rewards per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Rewards")
    ax1.legend()

    # Plot steps per episode for all samples
    ax2 = plt.subplot(gs[0, 1])
    for j, sample in enumerate(samples):
      ax2.plot(sample['steps'], label=f"Sample {j + 1}")
    ax2.set_title(f"Steps per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.legend()

    # Plot value function for all samples as a line plot
    ax3 = plt.subplot(gs[1, :])
    for j, sample in enumerate(samples):
      ax3.plot(sample['value_function'], label=f"Sample {j + 1}")
    ax3.set_title(f"Value Function")
    ax3.set_xlabel("State Index")
    ax3.set_ylabel("Predicted Value")
    ax3.legend()

    plt.tight_layout()
    plt.show()


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
