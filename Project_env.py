import numpy as np
from copy import deepcopy


class BoardGame:

    def __init__(self):
        self.n = 7
        self.start_state = np.array([
            [0, 0, 0, 7, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 7, 0, 0, 0, 0]
        ])
        self.p1_start = 3
        self.p1_end = 2
        self.p2_start = 19
        self.p2_end = 18
        self.state = self.start_state

        self.turn = 0
        self.start_board = np.array([
                [0, 0, 0, self.n, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, self.n, 0, 0, 0, 0, 0]
        ])
        self.board = self.start_board

        self.reward = 0
        self.rosettes = {0: (1, 7), 1: (4, ), 2: (1, 7)}
        self.grid = self.create_grid()

    def encode_state(self, turn, board):
        # turn = self.turn
        turn = int(turn)
        p1 = board[0, 2]
        p2 = board[2, 2]
        # p1_blue = np.concatenate((self.board[0, 4:], self.board[0, :2]))
        # p2_blue = np.concatenate((self.board[2, 4:], self.board[2, :2]))
        p1_blue = ''.join(board[0, self.blue_mask])
        p2_blue = ''.join(board[2, self.blue_mask])

        green = ''.join(board[1: 0])

        green_bin = np.binary_repr(int(green, 3))
        p1_bin = np.binary_repr(p1)
        p2_bin = np.binary_repr(p2)

        p1_bin = '0'*(3-len(p1_bin)) + p1_bin
        p2_bin = '0'*(3-len(p2_bin)) + p2_bin
        green_bin = '0'*(13-len(green_bin)) + green_bin

        bit_string = f'{turn}{p1_bin}{p2_bin}{p1_blue}{p2_blue}{green_bin}'
        print(bit_string)
        assert len(bit_string) == 32, f'{len(bit_string)} {bit_string}'

        return int(bit_string)

    def decode_state(self, state):
        # bit_string = np.binary_repr(state)
        # bits = np.array(list(bit_string))
        bits = np.binary_repr(state)
        bits = '0'*(32-len(bits)) + bits

        turn = int(bits[0])
        p1_score = int(bits[1:3])
        p2_score = int(bits[3:6])
        p1_blue = bits[7:13]
        p2_blue = bits[13:19]
        green = np.base_repr(int(bits[19:]), 3)
        green = '0'*(8-len(green)) + green

        board = np.array([
            [*list(p1_blue[:2]), p1_score, self.n - p1_score, *list(p1_blue[2:])],
            list(green),
            [*list(p2_blue[:2]), p2_score, self.n - p2_score, *list(p2_blue[2:])]
        ])
        print(board.shape)

        return turn, board

    def create_grid(self):
        grid = {}
        i = 0
        for j in range(self.start_state.shape[1]):
            grid[i] = (0, j)
            i += 1
        for j in reversed(range(self.start_state.shape[1])):
            grid[i] = (1, j)
            i += 1
        for j in range(self.start_state.shape[1]):
            grid[i] = (2, j)
            i += 1

        return grid

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

    def transition(self, previous_state, action, roll, player_turn=1):
        """
        Performs the transition to the next state
        """

        current_row, current_column = self.grid[action]

        if player_turn == 1:
            if action + roll > 15:
                next_square_index = action + roll - 16
            else:
                next_square_index = action + roll
        else:
            if action + roll > 23:
                next_square_index = action + roll - 16
            else:
                next_square_index = action + roll

        next_row, next_column = self.grid[next_square_index]

        next_state = previous_state.copy()
        if player_turn == 1:
            if next_state[next_row, next_column] == 2 and next_square_index != self.p1_end:
                next_state[2, 3] += 1
            if next_square_index != self.p1_end:
                next_state[next_row, next_column] = 1
        else:
            if next_state[next_row, next_column] == 1 and next_square_index != self.p2_end:
                next_state[0, 3] += 1
            if next_square_index != self.p2_end:
                next_state[next_row, next_column] = 2

        if action == self.p1_start or action == self.p2_start:
            next_state[current_row, current_column] -= 1
        elif next_square_index == self.p1_end or next_square_index == self.p2_end:
            next_state[next_row, next_column] += 1
            next_state[current_row, current_column] = 0
        else:
            next_state[current_row, current_column] = 0

        next_turn = int(not bool(player_turn))
        return next_state, next_turn

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
