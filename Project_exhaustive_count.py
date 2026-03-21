import numpy as np
from Project_env import BoardGame

def main():
    env = BoardGame(2)
    turn = 0

    frontier = set([0])
    explored = set()

    while frontier:
        # encoded_state = frontier[0]
        f = iter(frontier) # if this works, it'll be funny
        encoded_state = next(f)
        frontier.remove(encoded_state)
        turn, current_state = env.decode_state(encoded_state)

        children = []
        for roll in [0, 1, 2, 3, 4]:
            actions = env.get_actions(current_state, turn, roll)
            for action in actions:
                next_turn, next_state = env.transition(current_state, action, roll, turn)
                next_encoded_state = env.encode_state(next_turn, next_state)
                if (next_encoded_state not in explored) and (next_encoded_state not in frontier):
                    # frontier.append(next_encoded_state)
                    frontier.add(next_encoded_state)


        if len(explored) % 1000 == 0:
            print(len(explored))
        explored.add(encoded_state)

    print(f'Explored: {len(explored)}')


if __name__ == "__main__":
    main()
