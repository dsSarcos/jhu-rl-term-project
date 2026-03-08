import numpy as np


class BoardGame:

    def __init__(self):
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
        self.reward = 0
        self.rosettes = {0: (1, 7), 1: (4, ), 2: (1, 7)}
        self.grid = self.create_grid()

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
        self.state = self.start_state

        return self.state

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

        return next_state

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
