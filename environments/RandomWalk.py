"""
MIT, based on Miguel Morales' implementation:
https://github.com/mimoralea/gym-walk
"""

import sys
import numpy as np
from six import StringIO
from string import ascii_uppercase
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pygame

import gymnasium as gym
from gymnasium import spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
WEST, EAST = 0, 1


def make_random_walk(n_states=19, p_stay=0.1, p_backward=0.4, max_blocks=50, render_mode=None, verbose=0):
  return WalkEnv(n_states=n_states, p_stay=p_stay, p_backward=p_backward, max_blocks=max_blocks, render_mode=render_mode, verbose=verbose)


class WalkEnv(gym.Env):
  metadata = {"render_modes": [None, "human", "ansi"], "render_fps": 5}

  def __init__(self, n_states=7, p_stay=0.0, p_backward=0.5, max_blocks=100, render_mode=None, verbose=0):
    # Set the verbosity level for logging messages; higher values may provide more detailed output.
    self.verbose = verbose

    # Specify the number of states in the environment, representing the discrete positions the agent can occupy.
    self.n_states = n_states

    # Define the number of possible actions the agent can take.
    # For this environment:
    # - 0 corresponds to moving backward.
    # - 1 corresponds to moving forward.
    self.n_actions = 2

    # Define the shape of the environment grid, which includes:
    # - The n_states representing the navigable positions.
    # - Two additional boundary states, one at each end of the grid, making the grid size n_states + 2.
    self.shape = (1, n_states + 2)

    # Determine the starting state index.
    # The agent starts at the center of the grid.
    self.start_state_index = self.shape[1] // 2

    # Assign a descriptive name to the environment.
    self.name = "RandomWalk"

    # Define the maximum number of blocks (steps) allowed during an episode.
    self.max_blocks = max_blocks

    # Initialize a counter to track the current step number within an episode.
    self.current_step = 0

    # Extract the number of rows and columns from the grid's shape.
    # Since this is a 1D environment, nrow will always be 1.
    self.nrow, self.ncol = self.shape

    # Define the action space as a discrete space with n_actions possible actions.
    # This informs reinforcement learning agents of the valid action set.
    self.action_space = spaces.Discrete(self.n_actions)

    # Define the observation space as a discrete space representing all possible grid positions.
    # The total number of positions is given by the product of the grid's dimensions (nrow × ncol).
    self.observation_space = spaces.Discrete(np.prod(self.shape))

    # Initialize the environment specification (env.spec) as None.
    # This is typically populated when creating an environment using gymnasium.make().
    self.spec = None

    # Create a placeholder for the environment's random number generator (RNG).
    # This RNG will be initialized later during the reset process with a specified or random seed.
    self.np_random = None

    self.render_mode = render_mode
    if render_mode == "human":
      pygame.init()
      self._pygame_initialized = True
      self.screen_size = (800, 200)
      self.cell_width = self.screen_size[0] // (self.shape[1] - 2)
      self.screen = pygame.display.set_mode(self.screen_size)
      pygame.display.set_caption("Random Walk")
      self.clock = pygame.time.Clock()
      self.screen.fill((0, 0, 0))
      pygame.display.flip()

    self.nS = nS = np.prod(self.shape)
    self.nA = nA = 2

    self.P = {}
    for s in range(nS):
      self.P[s] = {}
      for a in range(nA):
        p_forward = 1.0 - p_stay - p_backward

        # Ensure state transitions stay within bounds
        s_forward = np.clip(s - 1 if a == WEST else s + 1, 0, nS - 1)
        s_backward = np.clip(s + 1 if a == WEST else s - 1, 0, nS - 1)

        # Rewards for transitions
        r_forward = 1.0 if s == nS - 2 and s_forward == nS - 1 else 0.0
        r_backward = 1.0 if s == nS - 2 and s_backward == nS - 1 else 0.0

        # Termination checks
        d_forward = (s >= nS - 2 and s_forward == nS - 1) or (s <= 1 and s_forward == 0)
        d_backward = (s >= nS - 2 and s_backward == nS - 1) or (s <= 1 and s_backward == 0)

        # Transition probabilities
        self.P[s][a] = [
            (p_forward, s_forward, r_forward, d_forward),
            (p_stay, s, 0.0, s in [0, nS - 1]),
            (p_backward, s_backward, r_backward, d_backward),
        ]

        # Validate probabilities sum to 1.0
        assert np.isclose(p_forward + p_stay + p_backward, 1.0), \
            f"Probabilities do not sum to 1.0 for state {s}, action {a}"

    self.isd = np.zeros(nS)
    self.isd[self.start_state_index] = 1.0
    self.lastaction = None

    self.action_space = spaces.Discrete(self.nA)
    self.observation_space = spaces.Discrete(self.nS)

    self.reset()

  def step(self, action):
    # Increment step counter
    self.current_step += 1

    # Ensure not a nan action (existing logic)
    if np.isnan(action):
      return self.s, 0.0, True, True, {"success": False}

    action = round(action)

    if not self.action_space.contains(action):
      return self.s, 0.0, True, True, {}

    transitions = self.P[self.s][action]
    i = categorical_sample([t[0] for t in transitions], self.np_random)
    p, s, reward, terminated = transitions[i]
    self.s = s
    self.lastaction = action

    # Check if maximum steps have been reached
    truncated = self.max_blocks is not None and self.current_step >= self.max_blocks
    terminated = terminated or truncated

    info = {"prob": p}
    if terminated and (self.s in [0, self.nS - 1]):
      info["success"] = True
    else:
      info["success"] = False

    return int(s), reward, terminated, truncated, info

  def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
    super().reset(seed=seed)
    self.s = int(categorical_sample(self.isd, self.np_random))
    self.lastaction = None
    self.current_step = 0  # Reset step counter
    return int(self.s), {"prob": self.isd[self.s]}

  def render(self):
    mode = self.render_mode
    if mode is None:
      return

    if mode == "ansi":
      outfile = StringIO() if mode == "ansi" else sys.stdout
      desc = np.asarray(["[" + ascii_uppercase[: self.shape[1] - 2] + "]"], dtype="c").tolist()
      desc = [[c.decode("utf-8") for c in line] for line in desc]
      color = "red" if self.s == 0 else "green" if self.s == self.nS - 1 else "yellow"
      desc[0][self.s] = utils.colorize(desc[0][self.s], color, highlight=True)
      outfile.write("\n")
      outfile.write("\n".join("".join(line) for line in desc) + "\n")
      return outfile.getvalue()

    if mode == "human":
      self.human()

  def human(self):
    if self.verbose > 0:
      print("Rendering the environment with Pygame...")

    # Colors
    BG_COLOR = (30, 30, 30)  # Background color
    CELL_COLOR = (200, 200, 200)  # Default cell color
    AGENT_COLOR = (255, 255, 0)  # Yellow for agent
    TERMINAL_COLOR = (0, 255, 0)  # Green for terminal states
    LEFT_TERMINAL_COLOR = (255, 0, 0)  # Red for left terminal state

    # Pump Pygame events to keep the event queue alive
    pygame.event.pump()

    # Clear the screen
    self.screen.fill(BG_COLOR)

    # Draw the grid
    for i in range(self.shape[1] - 2):
      cell_x = i * self.cell_width
      cell_rect = pygame.Rect(cell_x, 50, self.cell_width, 100)

      if i == 0:    # Left terminal state
        pygame.draw.rect(self.screen, LEFT_TERMINAL_COLOR, cell_rect)
      elif i == self.nS - 1:    # Right terminal state
        pygame.draw.rect(self.screen, TERMINAL_COLOR, cell_rect)
      elif i == self.s:    # Agent position
        pygame.draw.rect(self.screen, AGENT_COLOR, cell_rect)
      else:
        pygame.draw.rect(self.screen, CELL_COLOR, cell_rect)

      # Draw the border
      pygame.draw.rect(self.screen, (0, 0, 0), cell_rect, 2)

    # Refresh display
    pygame.display.flip()

    # Maintain frame rate
    self.clock.tick(self.metadata["render_fps"])


def plot_transition_matrix(env):
    # Adjust for terminal states
  n_states = env.shape[1] - 2  # Exclude start and terminal states
  n_actions = env.n_actions
  transition_probs = np.zeros((n_states, n_actions, n_states))
  rewards = np.zeros((n_states, n_actions, n_states))

  for s in range(n_states):
    for a in range(n_actions):
      for prob, next_s, reward, terminated in env.P[s + 1][a]:  # Offset to skip start state
        if 0 <= next_s - 1 < n_states:  # Adjust indices for valid range
          transition_probs[s, a, next_s - 1] += prob
          rewards[s, a, next_s - 1] += reward

  # Plot transition probabilities for each action
  fig, axes = plt.subplots(1, n_actions, figsize=(15, 5))
  for a in range(n_actions):
    sns.heatmap(
        transition_probs[:, a, :],
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        ax=axes[a]
    )
    axes[a].set_title(f"Action {a}: Transition Probabilities")
    axes[a].set_xlabel("Next State")
    axes[a].set_ylabel("Current State")

  plt.tight_layout()
  plt.show()


def plot_rewards(env):
    # Adjust for terminal states
  n_states = env.shape[1] - 2  # Exclude start and terminal states
  n_actions = env.n_actions
  rewards = np.zeros((n_states, n_actions, n_states))

  for s in range(n_states):
    for a in range(n_actions):
      for _, next_s, reward, _ in env.P[s + 1][a]:  # Offset to skip start state
        if 0 <= next_s - 1 < n_states:  # Adjust indices for valid range
          rewards[s, a, next_s - 1] = reward

  # Aggregate and plot rewards
  aggregated_rewards = rewards.sum(axis=1)  # Aggregate rewards over actions
  plt.figure(figsize=(10, 6))
  sns.heatmap(
      aggregated_rewards,
      annot=True,
      fmt=".2f",
      cmap="YlGnBu",
      cbar=True
  )
  plt.title("Rewards Matrix (Aggregated over Actions)")
  plt.xlabel("Next State")
  plt.ylabel("Current State")
  plt.show()


def estimate_goal_probability(env, num_simulations=100000):
  """
  Estimate the probability of reaching the goal in a random walk MDP.

  Parameters:
  - env: The random walk environment.
  - num_simulations: Number of simulations to run.

  Returns:
  - success_probability: Estimated probability of reaching the goal.
  """
  successes = 0

  for _ in range(num_simulations):
    env.reset()
    done = False

    while not done:
      action = env.action_space.sample()  # Take a random action
      _, _, terminated, truncated, info = env.step(action)
      done = terminated or truncated

      if terminated and info.get("success", False):
        successes += 1
        break

  success_probability = successes / num_simulations
  print(f"Estimated probability of reaching the goal: {success_probability:.4f}")


if __name__ == "__main__":
  env = make_random_walk(n_states=19, p_stay=0.0, p_backward=0.5, max_blocks=50)
  estimate_goal_probability(env)
  plot_transition_matrix(env)
  plot_rewards(env)
