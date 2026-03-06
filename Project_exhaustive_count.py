import numpy as np
from Project_env import BoardGame

def main():
    env = BoardGame()

    frontier = [env.state]
    explored = set()

    while frontier:
        current_state = frontier.pop(0)

        children = []
        for roll in [0, 1, 2, 3, 4]:
            actions = env.get_actions(current_state, roll)
            for action in actions:
                next_state = env.transition(current_state, action, roll)
                if (next_state not in explored) and (next_state not in frontier):
                    frontier.append(next_state)


        if len(explored) % 10 == 0:
            print(len(explored))
        explored.add(current_state)

    print(len(explored))
if __name__ == "__main__":
    main()
