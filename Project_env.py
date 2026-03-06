import numpy as np
from copy import deepcopy


class BoardGame:

    def __init__(self, n=7):
        if n > int('111', 2):
            raise ValueError
        self.n = n
        self.start_state = 0
        self.state = self.start_state

        self.turn = 0
        self.start_board = np.array([
                [0, 0, 0, self.n, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, self.n, 0, 0, 0, 0, 0]
        ])
        self.board = self.start_board

        self.reward = 0
        # self.rosettes = set([4, 8, 14])
        self.rosettes = set([(0, 1), (0, 7), (1, 4), (2, 1), (2, 7)])

        self.blue_mask = [4, 5, 6, 7, 0, 1]

        self.lane_masks = [
            [(0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (0, 0), (0, 1)],
            [(2, 4), (2, 5), (2, 6), (2, 7), (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1)],
        ]

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

    def get_actions(self, state, roll):
        turn, board = self.decode_state(state)
        scores = (board[0, 2], board[2, 2])

        # p1_blue = board[0, self.blue_mask]
        # p2_blue = board[2, self.blue_mask]
        # green = board[1]

        player = turn + 1

        # lane = np.concatenate((p1_blue[:4], green, p1_blue[4:])) \
            # if player ==1 else np.concatenate((p2_blue[:4], green, p2_blue[4:]))

        lane = board[self.lane_masks[turn]]

        positions = lane[lane == player]

        options = []
        for pos in positions:
            new_pos = pos + roll
            if new_pos < 12 and lane[new_pos] != player:
                options.append(pos)
            elif new_pos == 12:
                options.append(new_pos)

        if scores[turn] != self.n:
            options.append(-1)

        return options

    def transition(self, state, action, roll):
        """
        Performs the transition to the next state
        """
        turn, board = self.decode_state(state)
        scores = (board[0, 2], board[2, 2])

        # p1_blue = board[0, self.blue_mask]
        # p2_blue = board[2, self.blue_mask]
        # green = board[1]

        player = turn + 1

        lane = board[self.lane_masks[turn]]

        new_pos = roll if action == -1 else action + roll

        if new_pos == 12:
            scores[turn] += 1
            lane[action] = 0
        else:
            if lane[new_pos] != 0:
                scores[not turn] -= 1
            lane[new_pos] = player

        turn = not turn

        # return board
        return self.encode_state(turn, board)

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
