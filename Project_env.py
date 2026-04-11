import numpy as np
from copy import deepcopy


class BoardGame:

    def __init__(self, n=7, print_states=False):
        self.print_states = print_states
        self.n = n

        self.p1_start = 3
        self.p1_end = 2
        self.rosettes = [3, 7, 13]
        self.p2_start = 19
        self.p2_end = 18

        self.p1_pieces = n
        self.p2_pieces = n

        self.p1_states = np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.p2_states = np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


        self.start_board = np.array([
                [0, 0, 0, self.n, 0, 0, 0, 0],
                [0, 0, 0,      0, 0, 0, 0, 0],
                [0, 0, 0, self.n, 0, 0, 0, 0]
        ])

        # self.rosettes = {0: (1, 7), 1: (4, ), 2: (1, 7)}
        self.rosettes = {(0, 7), (0, 1), (1, 4), (2, 7), (2, 1)}
        self.rosettes_lane = {4, 8, 14}
        self.grid, self.inverse_grid = self.create_grid()
        self.rolls = []

    def encode_state(self, turn, board, roll):
        rolls = np.binary_repr(roll)
        turn = int(turn)
        p1 = board[0, 2]
        p2 = board[2, 2]
        p1_blue = ''.join(board[0, [4, 5, 6, 7, 0, 1]].astype('<U1'))
        p2_blue = ''.join(board[2, [4, 5, 6, 7, 0, 1]].astype('<U1'))

        green = ''.join(board[1].astype('<U1'))

        green_bin = np.binary_repr(int(green, 3))
        p1_bin = np.binary_repr(p1)
        p2_bin = np.binary_repr(p2)

        p1_bin = '0'*(3-len(p1_bin)) + p1_bin
        p2_bin = '0'*(3-len(p2_bin)) + p2_bin
        green_bin = '0'*(13-len(green_bin)) + green_bin

        bit_string = f"{int(turn)}{rolls}{p1_bin}{p2_bin}{p1_blue}{p2_blue.replace('2','1')}{green_bin}"
        # print(f'binary: {bit_string}')
        # assert len(bit_string) == 37, f'{len(bit_string)} {bit_string}'

        return int(bit_string, 2)

    def decode_state(self, state):
        # print("DECODE")
        bits = np.binary_repr(state)
        bits = '0'*(32-len(bits)) + bits

        turn = int(bits[0])
        p1_score = int(bits[1:3], 2)
        p2_score = int(bits[3:6], 2)
        p1_blue = bits[7:13]
        p2_blue = bits[13:19].replace('1', '2')
        green = np.base_repr(int(bits[19:], 2), 3)
        green = '0'*(8-len(green)) + green

        board = np.array([
            [*list(p1_blue[:2]), p1_score, self.n - p1_score, *list(p1_blue[2:])],
            list(green),
            [*list(p2_blue[:2]), p2_score, self.n - p2_score, *list(p2_blue[2:])]
        ]).astype(int)

        scores = {
            1: p1_score,
            2: p2_score
        }
        for player, score in scores.items():
            idx = player if player == 2 else 0
            lane_mask = [(idx, 4), (idx, 5), (idx, 6), (idx, 7),
                    (1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0),
                    (idx, 0), (idx, 1)]
            
            lane = np.array([board[x] for x in lane_mask])
            board[idx, 3] = self.n - scores[player] - (lane == player).sum()
        # print(board)

        return turn, board

    def create_grid(self):
        grid = {}
        grid_inv = {}
        i = 0
        for j in range(self.start_board.shape[1]):
            grid[i] = (0, j)
            grid_inv[(0, j)] = i
            i += 1
        for j in reversed(range(self.start_board.shape[1])):
            grid[i] = (1, j)
            grid_inv[(1, j)] = i
            i += 1
        for j in range(self.start_board.shape[1]):
            grid[i] = (2, j)
            grid_inv[(2, j)] = i
            i += 1

        return grid, grid_inv

    def execute_action(self, previous_state, action, player_turn=0, roll=np.random.choice([0, 1, 2, 3, 4], p=[1/16, 1/4, 3/8, 1/4, 1/16])):
        """
        Interface: execute_action(previous_state, action, turn)

        Execute the action given by action. Causes the next state to be
        determined, the state of the environment to be updated, and the
        applicable reward to be calculated. Returns the new state, the reward, and the
        terminal flag (in that order)
        """
        self.p2_rosette = False
        reward = 0
        next_turn, next_state = self.transition(previous_state, action, roll, player_turn=0)
        if self.get_terminal_flag(next_state) is True:
            reward = 1
        else:
            if next_turn == 1:
                next_state, reward, _ = self.play_turn(next_state)

        return next_state, reward, self.get_terminal_flag(next_state)

    def play_turn(self, previous_state, reward=0):
        self.p2_rosette = False
        roll = np.random.choice([0, 1, 2, 3, 4], p=[1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])
        p2_actions = self.get_actions(previous_state, 1, roll)
        if p2_actions:
            action = np.random.choice(p2_actions)
            _, next_state = self.transition(previous_state, action, roll, player_turn=1)
            if self.get_terminal_flag(next_state) is True:
                reward = -1
        else:
            next_state = previous_state

        return next_state, reward, self.get_terminal_flag(next_state)

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

    def transition(self, previous_state, action, roll, player_turn=0):
        """
        Performs the transition to the next state
        """

        current_row, current_column = self.grid[action]

        if player_turn == 0:
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

        if player_turn == 0:
            if previous_state[next_row, next_column] == 2 and next_square_index != self.p1_end:
                previous_state[2, 3] += 1
            if next_square_index != self.p1_end:
                previous_state[next_row, next_column] = 1
        else:
            if previous_state[next_row, next_column] == 1 and next_square_index != self.p2_end:
                previous_state[0, 3] += 1
            if next_square_index != self.p2_end:
                previous_state[next_row, next_column] = 2

        if action == self.p1_start or action == self.p2_start:
            previous_state[current_row, current_column] -= 1
        elif next_square_index == self.p1_end or next_square_index == self.p2_end:
            previous_state[next_row, next_column] += 1
            previous_state[current_row, current_column] = 0
        else:
            previous_state[current_row, current_column] = 0

        next_turn = player_turn
        if (next_row, next_column) not in self.rosettes: 
            next_turn = int(not player_turn) 
        else:
            if player_turn == 1:
                self.p2_rosette = True

        return next_turn, previous_state

    def get_terminal_flag(self, board):
        """
        Returns true if the current state is a terminal state, and false otherwise
        """
        if board[0, 2] == self.n or board[2, 2] == self.n:
            return True
        else:
            return False

    def reset(self):
        self.start_board[:] = 0.0
        self.start_board[0, 3] = self.n
        self.start_board[2, 3] = self.n
