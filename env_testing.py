from Project_env import BoardGame
import numpy as np

if __name__ == "__main__":
    ''' Index assigned to each element in the 2D array
        [[ 0,  1,  2,  3,  4,  5,  6,  7],
         [15, 14, 13, 12, 11, 10,  9,  8],
         [16, 17, 18, 19, 20, 21, 22, 23]]
    '''
    env = BoardGame()
    # Player 1 Moves
    # Move 1
    state = np.array(
        [[0, 0, 0, 7, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 7, 0, 0, 0, 0]]
    )
    action = 3
    roll = 4
    np.testing.assert_array_equal(env.transition(state, action, roll, player_turn=1), np.array(
        [[0, 0, 0, 6, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 7, 0, 0, 0, 0]]
    ))
    # Move 2
    state = np.array(
        [[0, 0, 0, 6, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 7, 0, 0, 0, 0]]
    )
    action = 7
    roll = 4
    np.testing.assert_array_equal(env.transition(state, action, roll, player_turn=1), np.array(
        [[0, 0, 0, 6, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 7, 0, 0, 0, 0]]
    ))
    # Move 3
    state = np.array(
        [[0, 0, 0, 6, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 7, 0, 0, 0, 0]]
    )
    action = 15
    roll = 3
    np.testing.assert_array_equal(env.transition(state, action, roll, player_turn=1), np.array(
        [[0, 0, 1, 6, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 7, 0, 0, 0, 0]]
    ))
    # Move 4
    state = np.array(
        [[0, 0, 1, 5, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 7, 0, 0, 0, 0]]
    )
    action = 14
    roll = 4
    np.testing.assert_array_equal(env.transition(state, action, roll, player_turn=1), np.array(
        [[0, 0, 2, 5, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 7, 0, 0, 0, 0]]
    ))

    # Player 2 Moves
    # Move 1
    state = np.array(
        [[0, 0, 0, 7, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 7, 0, 0, 0, 0]]
    )
    action = 19
    roll = 4
    np.testing.assert_array_equal(env.transition(state, action, roll, player_turn=2), np.array(
        [[0, 0, 0, 7, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 6, 0, 0, 0, 2]]
    ))
    # Move 2
    state = np.array(
        [[0, 0, 0, 7, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 6, 0, 0, 0, 2]]
    )
    action = 23
    roll = 4
    np.testing.assert_array_equal(env.transition(state, action, roll, player_turn=2), np.array(
        [[0, 0, 0, 7, 0, 0, 0, 0],
         [0, 0, 0, 0, 2, 0, 0, 0],
         [0, 0, 0, 6, 0, 0, 0, 0]]
    ))
    # Move 3
    state = np.array(
        [[0, 0, 0, 7, 0, 0, 0, 0],
         [2, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 6, 0, 0, 0, 0]]
    )
    action = 15
    roll = 3
    np.testing.assert_array_equal(env.transition(state, action, roll, player_turn=2), np.array(
        [[0, 0, 0, 7, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 6, 0, 0, 0, 0]]
    ))
    # Move 4
    state = np.array(
        [[0, 0, 0, 7, 0, 0, 0, 0],
         [0, 2, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 5, 0, 0, 0, 0]]
    )
    action = 14
    roll = 4
    np.testing.assert_array_equal(env.transition(state, action, roll, player_turn=2), np.array(
        [[0, 0, 0, 7, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 2, 5, 0, 0, 0, 0]]
    ))

    # Capturing
    # Move 1
    state = np.array(
        [[0, 0, 0, 6, 0, 0, 0, 0],
         [0, 2, 0, 1, 0, 0, 0, 0],
         [0, 0, 1, 5, 0, 0, 0, 0]]
    )
    action = 12
    roll = 2
    np.testing.assert_array_equal(env.transition(state, action, roll, player_turn=1), np.array(
        [[0, 0, 0, 6, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 6, 0, 0, 0, 0]]
    ))

    # Move 2
    state = np.array(
        [[0, 0, 0, 6, 0, 0, 0, 0],
         [0, 1, 0, 2, 0, 0, 0, 0],
         [0, 0, 1, 5, 0, 0, 0, 0]]
    )
    action = 12
    roll = 2
    np.testing.assert_array_equal(env.transition(state, action, roll, player_turn=2), np.array(
        [[0, 0, 0, 7, 0, 0, 0, 0],
         [0, 2, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 5, 0, 0, 0, 0]]
    ))