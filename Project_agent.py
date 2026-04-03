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
    def get_prime_action(actions, action_values, actions_indices):
        prime_action = np.argmax(action_values)
        prime_value = action_values[prime_action]

        prime_actions = action_values[action_values == prime_value]
        prob = 1/len(prime_actions)

        prime_action_probs = []
        for i in range(len(actions)):
            if i in actions_indices and actions[i] == prime_value:
                prime_action_probs.append(prob)
            else:
                prime_action_probs.append(0.0)

        return np.random.choice(len(actions), p=prime_action_probs)

    def select_action(self, state, actions_indices):
        action_value_array = self.q_table[state]
        available_action_values = action_value_array[actions_indices]

        prime_action = self.get_prime_action(action_value_array, available_action_values, actions_indices)
        all_actions = [i for i in range(len(available_action_values)) if i != prime_action]

        self.prob_prime = 1 - self.eps + (self.eps / len(actions_indices))
        self.prob_sub_prime = self.eps / len(actions_indices)

        action_probs = [self.prob_sub_prime for _ in range(len(all_actions))]
        action_probs.append(self.prob_prime)

        all_actions.append(prime_action)

        self.action = np.random.choice(len(all_actions), p=action_probs)

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
                actions_indices = environment.get_actions()
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

    def play_episodes(self, environment, number_of_episodes):
        for i in range(0, number_of_episodes):
            reward, game_end, current_state = np.nan, False, environment.get_state()
            actions_indices = environment.get_actions()
            current_action = self.select_action(current_state, actions_indices)

            while not game_end:
                actions_indices = environment.get_actions()
                next_action = self.select_action(current_state, actions_indices)

                self.update_trajectory_table(reward, current_state, current_action)
                next_state, reward, game_end = environment.execute_action(current_action)
                if self.enable_learning is True:
                    self.update_q_table(current_state, current_action, reward, next_state, next_action)
                current_state = next_state
                current_action = next_action

            self.update_trajectory_table(reward, current_state, np.nan)
            self.calculate_episode_return(i)
            self.clear_trajectory_table()
            environment.reset()
