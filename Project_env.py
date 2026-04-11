import numpy as np
from copy import deepcopy


class BoardGame:

    def __init__(self, n=7, print_states=False):
        self.print_states = print_states
        self.n = n

        self.rosettes = {4, 8, 14}
        self.p2_start = 19
        self.p2_end = 18

        self.p1_pieces = self.n
        self.p2_pieces = self.n

        self.p1_states = np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.p2_states = np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def encode_state(self, ai, bi, roll):
        a = ai[1:]
        b = bi[1:]

        a_bits = "".join(map(str, a))
        b_bits = "".join(map(str, b))

        state_bits = a_bits + b_bits

        roll_bits = f"{roll:03b}"

        bit_string = roll_bits + state_bits

        return int(bit_string, 2)

    def execute_action(self, previous_state, action, player_turn=0, roll=np.random.choice([0, 1, 2, 3, 4], p=[1/16, 1/4, 3/8, 1/4, 1/16])):
        """
        Interface: execute_action(previous_state, action, turn)

        Execute the action given by action. Causes the next state to be
        determined, the state of the environment to be updated, and the
        applicable reward to be calculated. Returns the new state, the reward, and the
        terminal flag (in that order)
        """
        reward = 0
        ai, bi = previous_state
        next_ai, next_bi, next_turn = self.transition(ai, bi, action, roll, False)
        if self.get_terminal_flag(next_ai, next_bi) is True:
            reward = 1
        else:
            if next_turn is True:
                return self.play_turn(next_bi, next_ai)

        return (ai, bi), reward, self.get_terminal_flag(ai, bi), next_turn

    def play_turn(self, ai, bi, reward=0):
        roll = np.random.choice([0, 1, 2, 3, 4], p=[1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])
        p2_actions = self.get_actions(ai, bi, roll)
        if p2_actions:
            action = np.random.choice(p2_actions)
            ai, bi, next_turn = self.transition(ai, bi, action, roll, True)
            if self.get_terminal_flag(ai, bi) is True:
                reward = -1
        else:
            next_turn = False

        return (bi, ai), reward, self.get_terminal_flag(ai, bi), next_turn

    def get_actions(self, ai, bi, roll):
        if roll == 0:
            return []

        a = ai[1:]
        b = bi[1:]

        if not np.any(a):
            return np.array([0])

        # Array of indices
        temp_a_indices = np.nonzero(a == 1)[0]

        # Add roll to current indices to find their corresponding new indices
        a_indices = temp_a_indices + roll

        # Remove any indices already occupied by current player
        a_indices = a_indices[np.isin(a_indices, temp_a_indices, invert=True)]

        # Remove any indices that do not roll out of the board by the exact amount
        a_indices = a_indices[a_indices <= 14]

        # Get opponent occupied indices
        b_indices = b == 1

        # Remove any indices that land on opponent's occupied florette
        if b_indices[7]:
            a_indices = a_indices[a_indices != 7]

        # Return original allowed indices from current player
        a_indices = a_indices - roll + 1

        if ai[0] != 0 and not ai[roll]:
            a_indices = np.insert(a_indices, 0, 0)

        return a_indices

    def transition(self, ai, bi, action, roll, player_turn):
        """
        Performs the transition to the next state
        """
        if action + roll == 15:
            ai[action] = 0
            return ai, bi

        ai[action] = 0
        ai[action + roll] = 1

        if 4 < action + roll <= 12:
            if bi[action + roll] == 1:
                bi[0] += 1
                bi[action + roll] = 0

        if action + roll not in self.rosettes:
            player_turn = not player_turn

        return ai, bi, player_turn

    def get_terminal_flag(self, ai, bi):
        """
        Returns true if the current state is a terminal state, and false otherwise
        """
        return ai.sum() == self.n or bi.sum() == self.n

    def reset(self):
        self.p1_states[:] = 0.0
        self.p2_states[:] = 0.0

        self.p1_states[0] = self.n
        self.p2_states[0] = self.n

