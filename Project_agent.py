import numpy as np
import random
import pandas as pd

from collections import defaultdict
from functools import partial
from copy import deepcopy


class RLAgent:

    def __init__(self,
                 eps = 0.1,
                 alpha = 0.1,
                 gamma = 1.0,
                 eps_min = None,
                 n_dec=-1,
                 ):
        
        self.temp_eps = eps
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
        self.eps = 0.0

    def enable(self):
        self.enable_learning = True
        self.eps = deepcopy(self.temp_eps)

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

    def e_greedy(self, actions, turn):
        # f = min if turn else max
        f = max
        a_star_idx = actions.index(f(actions))
        if len([x for x in actions if x == a_star_idx]):
            a_star_idx = self.get_prime_action(actions, a_star_idx)

        eps = self.eps
        if self.eps_min:
            r = f((self.n_dec-self.n_step)/self.n_dec, 0)
            eps = (self.eps - self.eps_min)*r + self.eps_min

        if eps <= random.random():
            return a_star_idx
        else:
            b = len(actions)
            idx = random.randint(0, b-1)
            return idx

    def select_action(self, state, action_indices, turn):
        # print("Turn = ", self.turn)
        self.state = state
        # print("State = ", self.state)
        # actions = self.q_table[state][action_indices]
        actions = [x for i, x in enumerate(self.q_table[state]) if i in action_indices]
        action = self.e_greedy(actions, turn)
        self.action = action_indices[action]
        return self.action

    def save_learning_table(self, file_name):
        print(f"Saving agent file: {file_name}")
        df = pd.DataFrame(data=self.q_table.values(), index=self.q_table.keys())
        # df.to_csv(file_name)
        df.to_parquet(file_name)

    def load_learning_table(self, file_name):
        print(f"Loading agent file: {file_name}")
        # df = pd.read_csv(file_name, index_col=0, low_memory=True)
        df = pd.read_parquet(file_name)
        self.q_table.update(df.T.to_dict(orient="list"))


class QLearner(RLAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_q_table(self, s, a, r, s_):
        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
                r + self.gamma * max(self.q_table[s_]) - self.q_table[s][a])

    def play_episodes(self, environment, n_step=None):
        current_turn = False
        self.n_step = n_step
        reward = 0
        game_end = False
        previous_sa = {
            0: None,
            1: None
        }
        while not game_end:
            roll = random.choices([0, 1, 2, 3, 4], weights=[1/16, 1/4, 3/8, 1/4, 1/16]).pop()
            current_state = (environment.p1_states, environment.p2_states) if not current_turn else (environment.p2_states, environment.p1_states)
            actions_indices = environment.get_actions(*current_state, roll)
            if actions_indices:
                encoded_state = environment.encode_state(*current_state)
                action = self.select_action(encoded_state, actions_indices, current_turn)
                previous_sa[current_turn] = (encoded_state, action)

                next_state, reward, game_end, next_turn = environment.execute_action(current_state, action, roll, current_turn)

                if current_turn == next_turn: # player has hit a rosette
                    self.update_q_table(
                        encoded_state,
                        action,
                        reward,
                        environment.encode_state(*next_state)
                    )
                    previous_sa[current_turn] = None
                else:
                    if previous_sa[next_turn]:
                        s, a = previous_sa[next_turn]
                        self.update_q_table(
                            s,
                            a,
                            reward,
                            environment.encode_state(*next_state)
                        )
                        previous_sa[next_turn] = None
                
                current_turn = next_turn
                current_state = next_state
            else:
                current_turn = not current_turn

        remaining_player, s, a = [(turn, val[0], val[1]) for turn, val in previous_sa.items() if val].pop()
        self.update_q_table(s, a, -reward, environment.encode_state(*next_state)) # NEXT STATE MIGHT BE WRONG HERE

        environment.reset()
        p1_return = 1 if previous_sa[0] is None else -1
        return p1_return, -p1_return


class eSARSA(RLAgent):

    def __init__(self, *args, on_policy=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_policy = on_policy

    def _get_probs(self, values):
        if self.on_policy:
            a_star_idx = values.index(max(values))
            A = len(values)

            if A == 1:
                probs = [1.0]
            else:
                probs = [1 - self.eps + (self.eps / A) if x == a_star_idx else self.eps / A for x in range(len(values))]

            return probs
        else:
            A = len(values)
            return [1/A for _ in values]

    def update_q_table(self, s, a, r, s_, environment):
        sum_actions = 0.
        encoded_next_state = environment.encode_state(*s_)
        for roll, prob in zip([1, 2, 3, 4], [1/4, 3/8, 1/4, 1/16]):
            actions = environment.get_actions(*s_, roll)
            if actions:
                values = [self.q_table[encoded_next_state][a] for a in actions]
                probs = self._get_probs(values)
                sum_actions += prob * sum([p * q for p, q in zip(probs, values)])

        self.q_table[s][a] = self.q_table[s][a] + self.alpha * (
                r + self.gamma * sum_actions - self.q_table[s][a])

    def play_episodes(self, environment, n_step=None):
        current_turn = False
        self.n_step = n_step
        reward = 0
        game_end = False
        previous_sa = {
            0: None,
            1: None
        }
        while not game_end:
            roll = random.choices([0, 1, 2, 3, 4], weights=[1/16, 1/4, 3/8, 1/4, 1/16]).pop()
            current_state = (environment.p1_states, environment.p2_states) if not current_turn else (environment.p2_states, environment.p1_states)
            actions_indices = environment.get_actions(*current_state, roll)
            if actions_indices:
                encoded_state = environment.encode_state(*current_state)
                action = self.select_action(encoded_state, actions_indices, current_turn)
                previous_sa[current_turn] = (encoded_state, action)

                next_state, reward, game_end, next_turn = environment.execute_action(current_state, action, roll, current_turn)

                if current_turn == next_turn: # player has hit a rosette
                    self.update_q_table(
                        encoded_state,
                        action,
                        reward,
                        next_state,
                        environment
                    )
                    previous_sa[current_turn] = None
                else:
                    if previous_sa[next_turn]:
                        s, a = previous_sa[next_turn]
                        self.update_q_table(
                            s,
                            a,
                            reward,
                            next_state,
                            environment
                        )
                        previous_sa[next_turn] = None
                
                current_turn = next_turn
                current_state = next_state
            else:
                current_turn = not current_turn

        remaining_player, s, a = [(turn, val[0], val[1]) for turn, val in previous_sa.items() if val].pop()
        self.update_q_table(s, a, -reward, next_state, environment) # NEXT STATE MIGHT BE WRONG HERE

        environment.reset()
        p1_return = 1 if previous_sa[0] is None else -1
        return p1_return, -p1_return

