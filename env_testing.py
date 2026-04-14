from Project_env import BoardGame
import numpy as np

if __name__ == "__main__":
    env = BoardGame()

    # Case start of game
    A = np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    B = np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    indices = env.get_actions(A, B, 3)
    np.testing.assert_array_equal(indices, np.array([0]))

    # Player one with a single piece at play
    A = np.array([6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    B = np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    indices = env.get_actions(A, B, 3)
    np.testing.assert_array_equal(indices, np.array([0, 1]))

    # Check for rosette from opponent
    A = np.array([6, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    B = np.array([6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    indices = env.get_actions(A, B, 1)
    np.testing.assert_array_equal(indices, np.array([0]))

    # Player with blocking piece
    A = np.array([5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    B = np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    indices = env.get_actions(A, B, 1)
    np.testing.assert_array_equal(indices, np.array([2]))

    # Player with blocking piece and out of board not by exact roll
    A = np.array([3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    B = np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    indices = env.get_actions(A, B, 2)
    np.testing.assert_array_equal(indices, np.array([1, 2, 13]))
