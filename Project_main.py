import Project_env
import Project_agent

from datetime import datetime

import numpy as np

if __name__ == "__main__":
    env = Project_env.BoardGame(n=4, print_states=False)
    agent = Project_agent.QLearner()

    # agent.disable()

    num_episodes = 10000
    start_time = datetime.now()
    agent.play_episodes(env, num_episodes)
    finish_time = datetime.now()
    np.savetxt('Project_experiment1.csv', agent.returns, delimiter=',')
    print(f"Average returns: {np.mean(agent.returns)}")
    print(f"Number of visited states: {len(agent.q_table.keys())}")
    print(f"Total time in minutes: {(finish_time - start_time).strftime('%H:%M:%S')}")
