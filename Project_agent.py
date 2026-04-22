import numpy as np
import random
import pandas as pd

from collections import defaultdict
from functools import partial


class RLAgent:

    def __init__(self,
                 eps = 0.1,
                 alpha = 0.1,
                 gamma = 1.0,
                 eps_min = None,
                 n_dec=-1,
                 seed=None
                 ):

        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.prob_prime = 0.0
        self.prob_sub_prime = 0.0
        self.q_table = defaultdict(lambda : [0]*15)
        self.returns = []
        self.n_step = -1
        self.n_dec = n_dec
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
        prime_actions = [i for i, x in enumerate(actions) if x == prime_value]

        return random.choice(prime_actions)

    def e_greedy(self, actions):
        a_star_idx = actions.index(max(actions))
        if len([x for x in actions if x == a_star_idx]):
            a_star_idx = self.get_prime_action(actions, a_star_idx)

        eps = self.eps
        if self.eps_min:
            r = max((self.n_dec-self.n_step)/self.n_dec, 0)
            eps = (self.eps - self.eps_min)*r + self.eps_min

        if eps <= random.random():
            return a_star_idx
        else:
            b = len(actions)
            idx = random.randint(0, b-1)
            return idx

    def select_action(self, state, action_indices):
        # print("Turn = ", self.turn)
        self.state = state
        # print("State = ", self.state)
        # actions = self.q_table[state][action_indices]
        actions = [x for i, x in enumerate(self.q_table[state]) if i in action_indices]
        action = self.e_greedy(actions)
        self.action = action_indices[action]
        return self.action

    def save_learning_table(self, file_name):
        print(f"Saving agent file: {file_name}")
        df = pd.DataFrame(self.q_table)
        df.to_csv(file_name, index=False)

    def load_learning_table(self, file_name):
        print(f"Loading agent file: {file_name}")
        df = pd.read_csv(file_name)
        df.columns = df.columns.astype(int)
        self.q_table.update(df.to_dict(orient='list'))


class QLearner(RLAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_q_table(self, s, a, r, s_):
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
                r + self.gamma * max(self.q_table[s_]) - self.q_table[s][a])

    def play_episodes(self, environment, n_step=None):
        current_turn = False
        # for i in range(0, number_of_episodes):
        self.n_step = n_step
        reward = 0
        game_end = False
        current_state = environment.p1_states, environment.p2_states
        while not game_end:
            roll = random.choices([0, 1, 2, 3, 4], weights=[1/16, 1/4, 3/8, 1/4, 1/16]).pop()
            actions_indices = environment.get_actions(*current_state, roll)
            if actions_indices and not current_turn:
                encoded_state = environment.encode_state(*current_state)
                action = self.select_action(encoded_state, actions_indices)

                next_state, reward, game_end, current_turn = environment.execute_action(current_state, action, roll=roll)
                if self.enable_learning is True:
                    self.update_q_table(encoded_state,
                                        action,
                                        reward,
                                        environment.encode_state(*next_state))
            else:
                next_state, reward, game_end, current_turn = environment.play_turn(*current_state[::-1])

            current_state = next_state

        # self.returns.append(reward)
        environment.reset()
        return reward
        # return self.rewards


class eSARSA(RLAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_probs(self, actions):
        prime_action = max(actions)
        # prime_actions = actions == prime_action
        prime_actions = [1 if x == prime_action else 0 for x in actions]
        n_prime_actions = sum(prime_actions)
        A = len(actions)

        probs = [self.eps / A if x == prime_action else 1 - self.eps + (self.eps / n_prime_actions) for x in actions]

        return probs

    def update_q_table(self, s, a, r, s_, environment):
        # TODO: Verify that probs add up to one
        sum_actions = 0.
        encoded_next_state = environment.encode_state(*s_)
        for roll, prob in zip([1, 2, 3, 4], [1/4, 3/8, 1/4, 1/16]):
            actions = environment.get_actions(*s_, roll)
            if actions:
                probs = self._get_probs(actions)
                sum_actions += prob * sum([p * q for p, q in zip(probs, self.q_table[encoded_next_state])])

        if self.q_table[s][a] > 0:
            print(self.q_table[s][a] + self.alpha * (
                r + self.gamma * sum_actions - self.q_table[s][a]))
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
                r + self.gamma * sum_actions - self.q_table[s][a])

    def play_episodes(self, environment, n_step=None):
        # expected SARSA needs the roll to be encoded outside of the state
        self.n_step = n_step
        current_turn = False
        reward = 0
        game_end = False
        current_state = environment.p1_states, environment.p2_states
        while not game_end:
            roll = random.choices([0, 1, 2, 3, 4], weights=[1/16, 1/4, 3/8, 1/4, 1/16]).pop()
            actions_indices = environment.get_actions(*current_state, roll)
            if actions_indices and not current_turn:
                encoded_state = environment.encode_state(*current_state)
                action = self.select_action(encoded_state, actions_indices)

                next_state, reward, game_end, current_turn = environment.execute_action(current_state, action, roll=roll)
                if self.enable_learning is True:
                    self.update_q_table(encoded_state,
                                        action,
                                        reward,
                                        next_state,
                                        environment)
            else:
                next_state, reward, game_end, current_turn = environment.play_turn(*current_state[::-1])

            current_state = next_state

        # self.returns.append(reward)
        environment.reset()
        return reward
