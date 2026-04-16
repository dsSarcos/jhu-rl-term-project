import numpy as np
import random
from copy import deepcopy
from collections import deque


class BoardGame:

    def __init__(self, n=7, print_states=False):
        self.print_states = print_states
        self.n = n

        self.rosettes = {4, 8, 14}
        self.p2_start = 19
        self.p2_end = 18

        self.p1_pieces = self.n
        self.p2_pieces = self.n

        # self.p2_states = [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # self.p1_states = [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.p1_states = deque([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.p2_states = deque([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.empty = []
        self.start = [0]

    def encode_state(self, ai, bi, roll):

        a1 = ai.popleft()
        b1 = bi.popleft()

        out = 0
        for i in ai:
            out = (out << 1) | i

        for i in bi:
            out = (out << 1) | i

        ai.appendleft(a1)
        bi.appendleft(b1)

        return out

    def execute_action(self, previous_state, action, player_turn=0, roll=random.choices([0, 1, 2, 3, 4], weights=[1/16, 1/4, 3/8, 1/4, 1/16]).pop()):
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
        if self.get_terminal_flag(next_ai, next_bi):
            reward = 1
        else:
            if next_turn is True:
                return self.play_turn(next_bi, next_ai)

        return (ai, bi), reward, self.get_terminal_flag(next_ai, next_bi), next_turn

    def play_turn(self, ai, bi, reward=0):
        roll = random.choices([0, 1, 2, 3, 4], weights=[1/16, 1/4, 3/8, 1/4, 1/16]).pop()
        p2_actions = self.get_actions(ai, bi, roll)
        if p2_actions:
            action = random.choice(p2_actions)
            ai, bi, next_turn = self.transition(ai, bi, action, roll, True)
            if self.get_terminal_flag(ai, bi):
                reward = -1
        else:
            next_turn = False

        return (bi, ai), reward, self.get_terminal_flag(ai, bi), next_turn

    def get_actions(self, ai, bi, roll):
        if roll == 0:
            return self.empty

        a1 = ai.popleft()
        b1 = bi.popleft()

        if not any(ai):
            ai.appendleft(a1)
            bi.appendleft(b1)
            return self.start

        # Array of indices
        temp_a_indices = [i for i in range(len(ai)) if ai[i] == 1]
        # print(temp_a_indices)

        # Remove any indices already occupied by current player
        a_indices = [x+roll for x in temp_a_indices if x+roll not in temp_a_indices and x+roll <= 14]


        # Get opponent occupied indices
        b_indices = [1 if x == 1 else 0 for x in bi]

        # Remove any indices that land on opponent's occupied florette
        if b_indices[7]:
            a_indices = [x for x in a_indices if x != 7]

        # Return original allowed indices from current player
        a_indices = [x - roll + 1 for x in a_indices]

        ai.appendleft(a1)
        bi.appendleft(b1)
        if ai[0] != 0 and not ai[roll]:
            a_indices = [0] + a_indices

        return a_indices

    def transition(self, ai, bi, action, roll, player_turn):
        """
        Performs the transition to the next state
        """
        if action + roll == 15:
            ai[action] = 0
            return ai, bi, not player_turn

        if action > 0:
            ai[action] = 0
        else:
            ai[action] -= 1

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
        return sum(ai) == 0 or sum(bi) == 0

    def reset(self):
        self.p1_states = deque([0] * 14)
        self.p2_states = deque([0] * 14)

        self.p1_states.appendleft(self.n)
        self.p2_states.appendleft(self.n)

