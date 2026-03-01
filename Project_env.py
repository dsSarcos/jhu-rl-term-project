import numpy as np


class BoardGame:

    def __init__(self):
        self.start_state = np.array([
            [0, 0, 7, 7, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 7, 7, 0, 0, 0, 0]
        ])
        self.state = self.start_state
        self.reward = 0
        self.rosettes = {0: (1, 7), 1: (4, ), 2: (1, 7)}

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

    def transition(self, previous_state, action):
        """
        Performs the transition to the next state
        """
        return

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
