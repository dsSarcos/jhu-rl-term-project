import Project_env
import Project_agent

if __name__ == "__main__":
    env = Project_env.BoardGame(print_states=False)
    agent = Project_agent.QLearner()

    agent.set_epsilon(0.1)
    agent.set_alpha(0.1)
    agent.set_gamma(1.0)

    num_episodes = 10
    agent.play_episodes(env, num_episodes)
    print(agent.extract_returns_per_episode())
