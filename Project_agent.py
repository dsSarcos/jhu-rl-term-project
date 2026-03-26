import numpy as np
from collections import defaultdict
from functools import partial


class RLAgent:

    def __init__(self,
                 eps_type="fixed"):

        self.eps_type = eps_type
        self.eps = 0.0
        self.alpha = 0.0
        self.gamma = 0.0
        self.state = 0
        self.action = 0
        self.prob_prime = 0.0
        self.prob_sub_prime = 0.0
        self.q_table = defaultdict(partial(np.zeros, shape=15))
        self.learning_returns = []
        self.trajectory_table = np.empty((0, 3))  # [r, s, a]
        self.N = 0
        self.n_step = -1
        self.eps_max = 0.0
        self.eps_min = 0.0
        self.enable_learning = True

    def disable(self):
        self.enable_learning = False

    def enable(self):
        self.enable_learning = True

    def set_gamma(self, gamma):
        self.gamma = gamma

    def get_gamma(self):
        return self.gamma

    def set_alpha(self, alpha):
        self.alpha = alpha

    def get_alpha(self):
        return self.alpha

    def set_epsilon(self, eps):
        self.eps = eps

    def get_epsilon(self):
        return self.eps

    def update_trajectory_table(self, r, s, a):
        self.trajectory_table = np.append(self.trajectory_table, np.array([[r, s, a]]), axis=0)

    def calculate_episode_return(self, episode):
        self.learning_returns[episode] = sum(self.trajectory_table[1:, 0])

    def clear_trajectory_table(self):
        self.trajectory_table = np.empty((0, 3))

    @staticmethod
    def get_prime_action(action_values):
        prime_action = np.argmax(action_values)
        prime_value = action_values[prime_action]
        prime_actions = [i for i in range(len(action_values)) if action_values[i] == prime_value]

        return np.random.choice(prime_actions)

    def select_action(self, state, actions_indices):
        actions = self.q_table[state]
        action_values = actions[actions_indices]
        prime_action = self.get_prime_action(action_values)
        other_actions = np.array([i for i in range(len(action_values)) if i != prime_action])

        self.prob_prime = 1 - self.eps + (self.eps / len(actions_indices))
        self.prob_sub_prime = self.eps / len(actions_indices)

        action_probs = [self.prob_prime, self.prob_sub_prime, self.prob_sub_prime, self.prob_sub_prime]
        actions = [prime_action, other_actions[0], other_actions[1], other_actions[2]]
        self.action = np.random.choice(actions, p=action_probs)

        return self.action


class QLearner(RLAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_q_table(self, s, a, r, s_):
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
                r + self.gamma * self.q_table[s_].max() - self.q_table[s][a])

    def play_episodes(self, environment, number_of_episodes):
        for i in range(0, number_of_episodes):
            reward, game_end, current_state = np.nan, False, environment.get_state()

            while not game_end:
                actions_indices = environment.get_state()
                action = self.select_action(current_state, actions_indices)

                self.update_trajectory_table(reward, current_state, action)
                next_state, reward, game_end = environment.execute_action(action)
                if self.enable_learning is True:
                    self.update_q_table(current_state, action, reward, next_state)
                current_state = next_state

            self.update_trajectory_table(reward, current_state, np.nan)
            self.calculate_episode_return(i)
            self.clear_trajectory_table()
            environment.reset()


class SARSA(RLAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_q_table(self, s, a, r, s_, a_):
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
                r + self.gamma * self.q_table[s_][a_] - self.q_table[s][a])

