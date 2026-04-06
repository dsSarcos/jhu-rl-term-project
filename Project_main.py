import Project_env
import Project_agent

import numpy as np

if __name__ == "__main__":
    env = Project_env.BoardGame(print_states=False)
    agent = Project_agent.QLearner()

    agent.set_epsilon(0.1)
    agent.set_alpha(0.1)
    agent.set_gamma(1.0)

    num_episodes = 100_000
    agent.play_episodes(env, num_episodes)
    print(f"Average returns: {np.mean(agent.returns)}")
    print(f"Number of visited states: {len(agent.q_table.keys())}")
