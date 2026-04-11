import numpy as np
from collections import defaultdict
from functools import partial


class RLAgent:

    def __init__(self,
                 eps = 0.1,
                 alpha = 0.1,
                 gamma = 1.0,
                 eps_min = None
                 ):

        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.prob_prime = 0.0
        self.prob_sub_prime = 0.0
        self.q_table = defaultdict(partial(np.zeros, shape=15))
        self.returns = []
        self.n_step = -1
        self.eps_min = eps_min
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

    @staticmethod
    def get_prime_action(actions, prime_action):
        prime_value = actions[prime_action]
        prob = 1/np.count_nonzero(actions == prime_value)

        prime_action_probs = []
        for action_value in actions:
            if action_value == prime_value:
                prime_action_probs.append(prob)
            else:
                prime_action_probs.append(0.0)

        return np.random.choice(len(actions), p=prime_action_probs)

    def e_greedy(self, actions):
        a_star_idx = np.argmax(actions)
        if np.count_nonzero(actions == actions[a_star_idx]) > 1:
            a_star_idx = self.get_prime_action(actions, a_star_idx)

        eps = self.eps
        if self.eps_min:
            r = max((300_000-self.n_step)/300_000, 0)
            eps = (self.eps - self.eps_min)*r + self.eps_min

        rng = np.random.default_rng()
        if eps <= rng.random():
            return a_star_idx
        else:
            b = actions.size
            idx = rng.integers(low=0, high=b)
            return idx

    def select_action(self, state, action_indices):
        # print("Turn = ", self.turn)
        self.state = state
        # print("State = ", self.state)
        actions = self.q_table[state][action_indices]
        action = self.e_greedy(actions)
        self.action = action_indices[action]
        return self.action


class QLearner(RLAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_q_table(self, s, a, r, s_):
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
                r + self.gamma * self.q_table[s_].max() - self.q_table[s][a])

    def play_episodes(self, environment, number_of_episodes):
        current_turn = False
        for i in range(0, number_of_episodes):
            self.n_step = i
            if i % 1000 == 0:
                print(f"Number of episodes played: {i}")
            reward = 0
            game_end = False
            current_state = environment.p1_states, environment.p2_states
            while not game_end:
                roll = np.random.choice([0, 1, 2, 3, 4], p=[1/16, 1/4, 3/8, 1/4, 1/16])
                actions_indices = environment.get_actions(*current_state, roll)
                if actions_indices.size and not current_turn:
                    encoded_state = environment.encode_state(*current_state, roll)
                    action = self.select_action(encoded_state, actions_indices)

                    next_state, reward, game_end, current_turn = environment.execute_action(current_state, action, roll=roll)
                    if self.enable_learning is True:
                        self.update_q_table(encoded_state,
                                            action,
                                            reward,
                                            environment.encode_state(*next_state, roll))
                else:
                    next_state, reward, game_end, current_turn = environment.play_turn(*current_state[::-1])

                current_state = next_state

            self.returns.append(reward)
            # print(len(self.returns))
            environment.reset()


class SARSA(RLAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_q_table(self, s, a, r, s_, a_):
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
                r + self.gamma * self.q_table[s_][a_] - self.q_table[s][a])

    def play_episodes(self, environment, number_of_episodes):
        for i in range(0, number_of_episodes):
            reward, game_end, current_state = np.nan, False, environment.start_board

            roll = np.random.choice([0, 1, 2, 3, 4], p=[1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])
            actions_indices = environment.get_actions(current_state, 0, roll)
            current_action = self.select_action(environment.encode_state(0, current_state), actions_indices)

            while not game_end:
                # Refactor into expected SARSA - Off Policy TD Learning
                roll = np.random.choice([0, 1, 2, 3, 4], p=[1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])
                actions_indices = environment.get_actions(current_state, 0, roll)
                next_action = self.select_action(current_state, actions_indices)

                self.update_trajectory_table(reward, current_state, current_action)
                next_state, reward, game_end = environment.execute_action(current_state, current_action)
                if self.enable_learning is True:
                    self.update_q_table(environment.encode_state(0, current_state),
                                        current_action,
                                        reward,
                                        next_state,
                                        next_action)
                current_state = next_state
                current_action = next_action

            self.update_trajectory_table(reward, current_state, np.nan)
            self.calculate_episode_return(i)
            self.clear_trajectory_table()
            environment.reset()
