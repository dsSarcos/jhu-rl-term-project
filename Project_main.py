import Project_env
import Project_agent

import time

import numpy as np

if __name__ == "__main__":
    env = Project_env.BoardGame(n=4, print_states=False)
    agent = Project_agent.QLearner()

    agent.disable()

    num_episodes = 100_000
    start_time = time.time()
    agent.play_episodes(env, num_episodes)
    finish_time = time.time()
    np.savetxt('Project_experiment1.csv', agent.returns, delimiter=',')
    print(f"Average returns: {np.mean(agent.returns)}")
    print(f"Number of visited states: {len(agent.q_table.keys())}")
    print(f"Total time in minutes: {(finish_time - start_time)/60}")
