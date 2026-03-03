import numpy as np
from copy import deepcopy


class BoardGame:

    def __init__(self, n=7):
        if n > int('111', 2):
            raise ValueError
        self.n = n
        self.start_state = self.encode_state()
        self.state = self.start_state

        self.reward = 0
        self.rosettes = set([4, 8, 14])

    def encode_state(self,
                     turn=0,
                     p1=0,
                     p2=0,
                     p1_blue='000000',
                     p2_blue='000000',
                     green='00000000'):
        out = turn
        out <<= 3
        out | p1
        out <<= 3
        out | p2
        out <<= 6
        out | int(p1_blue, 2)
        out <<= 6
        out | int(p2_blue, 2)
        out <<= 13
        out | int(green, 3)

        self.state = out
        return out

    def _to_ternary(self, n):
        if n == 0:
            return '0' * 13
        out = []
        while n:
            n, r = divmod(n, 3)
            out.append(str(r))
        while len(out) < 13:
            out.append(0)
        return ''.join(reversed(out))

    def decode_state(self, state):
        green = state & int('1111111111111', 2)
        state >>= 13
        p2_blue = format(state & int('111111', 2), 'b')
        while len(p2_blue) < 6:
            p2_blue += '0'
        state >>= 6
        p1_blue = format(state & int('111111', 2), 'b')
        while len(p1_blue) < 6:
            p1_blue += '0'
        state >>= 6
        p2 = state & int('111', 2)
        state >>= 3
        p1 = state & int('111', 2)
        state >>= 3
        turn = state

        return turn, p1, p2, p1_blue, p2_blue, self._to_ternary(green)

    def get_actions(self, state, roll):
        turn, p1_score, p2_score, p1_blue, p2_blue, green = self.decode_state(state)
        scores = [p1_score, p2_score]
        player = turn + 1

        state_arr = []
        if not turn:
            state_arr = p1_blue[:4] + green + p1_blue[4:]
        else:
            state_arr = p2_blue[:4] + green + p2_blue[4:]

        options = [] # indices of current player's pieces
        for i in range(len(state_arr)):
            if 5 <= i <= 12:
                if state_arr[i] == player:
                    options.append(i+1)
            else:
                if state_arr[i] == 1:
                    options.append(i+1)

        if scores[turn] < self.n:
            options.append(0)

        return options

    def transition(self, state, action, roll):
        """
        Performs the transition to the next state
        """
        turn, p1_score, p2_score, p1_blue, p2_blue, green = self.decode_state(state)
        player = turn + 1
        scores = [p1_score, p2_score]
        new_pos = action + roll

        state_arr = []
        if not turn:
            state_arr = list(p1_blue[:4] + green + p1_blue[4:])
        else:
            state_arr = list(p2_blue[:4] + green + p2_blue[4:])

        if new_pos <= len(state_arr):
            if new_pos == len(state_arr):
                scores[turn] += 1
                state_arr[action-1] = '0'
            else:
                if action == 0:
                    state_arr[new_pos-1] = '1'
                else:
                    state_arr[action-1], state_arr[new_pos-1] = '0', '1'
                    if 5 <= new_pos <= 12:
                        state_arr[new_pos-1] = f'{player}'

        if not turn:
            p1_blue, temp = state_arr[:4], state_arr[4:]
            green, p1_blue_end = temp[:8], temp[8:]
            p1_blue += p1_blue_end
        else:
            p2_blue, temp = state_arr[:4], state_arr[4:]
            green, p2_blue_end = temp[:8], temp[8:]
            p2_blue += p2_blue_end

        if new_pos not in self.rosettes:
            turn = not turn

        return self.encode_state(turn, p1_score, p2_score, ''.join(p1_blue), ''.join(p2_blue), ''.join(green))

    def reset(self):
        """
        Resets the environment to the beginning of the episode.
        """
        self.board_state = deepcopy(self.start_state)

        return self.board_state

    def execute_action(self, action):
        """
        Execute the action given by action. Causes the next state to be
        determined, the state of the environment to be updated, and the
        applicable reward to be calculated. Returns the new state, the reward, and the
        terminal flag (in that order)
        """

        return self.state, self.reward, self.get_terminal_flag()

    def set_reward(self):
        """
        Sets the reward
        """
        if self.state[0, 3] == 7:
            self.reward = 1
        elif self.state[2, 3] == 7:
            self.reward = -1
        else:
            self.reward = 0

    def set_state(self, state):
        """
        Permits the state value to be set
        """
        self.state = state

    def get_terminal_flag(self):
        """
        Returns true if the current state is a terminal state, and false otherwise
        """
        if self.state[0, 3] == 7 or self.state[2, 3] == 7:
            return True
        else:
            return False
