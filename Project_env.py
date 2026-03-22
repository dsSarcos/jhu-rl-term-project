import numpy as np
from copy import deepcopy


class BoardGame:

    def __init__(self, n=7):
        self.n = n
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
        self.board = self.start_board.copy()

        self.reward = 0
        self.rosettes = {0: (1, 7), 1: (4, ), 2: (1, 7)}
        self.grid, self.inverse_grid = self.create_grid()

    def encode_state(self, turn, board):
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

        bit_string = f'{int(turn)}{p1_bin}{p2_bin}{p1_blue}{p2_blue.replace('2','1')}{green_bin}'
        # print(f'binary: {bit_string}')
        assert len(bit_string) == 32, f'{len(bit_string)} {bit_string}'

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
        for j in range(self.start_state.shape[1]):
            grid[i] = (0, j)
            grid_inv[(0, j)] = i
            i += 1
        for j in reversed(range(self.start_state.shape[1])):
            grid[i] = (1, j)
            grid_inv[(1, j)] = i
            i += 1
        for j in range(self.start_state.shape[1]):
            grid[i] = (2, j)
            grid_inv[(2, j)] = i
            i += 1

        return grid, grid_inv

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

    def get_actions(self, state, turn, roll):
        if roll == 0:
            return []
        # print(state)
        player = int(turn) + 1
        idx = player if player == 2 else 0

        starting = (idx, 3)
        lane = [(idx, 4), (idx, 5), (idx, 6), (idx, 7),
                (1, 7), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2), (1, 1), (1, 0),
                (idx, 0), (idx, 1)]

        options = []
        if state[starting] > 0 and state[lane[roll - 1]] != player:
            options.append(starting)

        for i, position in enumerate(lane):
            if state[position] == player:
                next_position = i + roll

                if next_position == len(lane):
                    options.append(position)
                elif next_position < len(lane) and state[lane[next_position]] != player:
                    options.append(position)

        return [self.inverse_grid[opt] for opt in options]

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

        next_state = previous_state.copy()
        if player_turn == 0:
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

        next_turn = int(not player_turn)
        return next_turn, next_state

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
