import numpy as np
from Sarsa.IHT import IHT, tiles
from Specs import AlgoSpecs
from ExpectedSarsa import ExpectedSarsa


class ExpectedSarsaTileCoding(ExpectedSarsa):
    def __init__(self, observation_space, action_space, alpha=0.01, epsilon=1.0, decay_rate=0.99, gamma=0.99, tilings=8, iht_size=4096, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_actions = action_space.n

        self.tilings = tilings
        self.iht = IHT(iht_size)
        self.weights = np.zeros(iht_size)
        
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.gamma = gamma
        

    def _get_features(self, state, action):
        state_list = state if isinstance(state, list) else state.tolist()
        return tiles(self.iht, self.tilings, state_list, [action])

    def _q_value(self, state, action):
        features = self._get_features(state, action)
        return np.sum(self.weights[features])

    def policy(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # Explore
        else:
            q_values = [self._q_value(state, action) for action in range(self.num_actions)]
            return np.argmax(q_values)  # Exploit

    def _update(self, state, action, reward, next_state, done):
        current_q = self._q_value(state, action)

        if done:
            target = reward
        else:
            next_action_probabilities = np.ones(self.num_actions) * (self.epsilon / self.num_actions)
            best_action = self.policy(next_state)
            next_action_probabilities[best_action] += (1.0 - self.epsilon)
            expected_q = np.dot(next_action_probabilities, [self._q_value(next_state, a) for a in range(self.num_actions)])

            target = reward + self.gamma * expected_q

        td_error = target - current_q
        features = self._get_features(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * td_error

    def predict(self, state):
        q_values = [self._q_value(state, action) for action in range(self.num_actions)]
        return np.argmax(q_values)

    def save(self, filename):
        np.savez(
            filename,
            weights=self.weights,
            params={
                "alpha": self.alpha,
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "tilings": self.tilings,
                "iht_size": len(self.weights),
            }
        )

    def load(self, filename):
        data = np.load(filename + ".npz", allow_pickle=True)
        self.weights = data["weights"]
        params = data["params"].item()
        self.alpha = params["alpha"]
        self.epsilon = params["epsilon"]
        self.gamma = params["gamma"]
        self.tilings = params["tilings"]
        self.iht = IHT(params["iht_size"])


if __name__ == "__main__":
    from Environments.RandomWalk import make_random_walk

    env = make_random_walk()

    agent = ExpectedSarsaTileCoding(
        observation_space=env.observation_space,
        action_space=env.action_space,
        alpha=0.01,
        epsilon=1.0,
        decay_rate=0.99,
        gamma=0.99,
        tilings=8,
        iht_size=4096,
    )

    num_episodes = 500
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

from CrossValidation import GridSearchCV


class ExpectedSarsaTCCV(GridSearchCV):
  def new(self, **params):
    return ExpectedSarsaTileCoding(
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
      "alpha": [0.003],
      "gamma": [0.2],
      "tilings": [8],
      "iht_size": [4096],
  }

  cv = ExpectedSarsaTCCV(env, param_grid)
  cv.search()

  estimate_goal_probability(env)
  cv.summary()
