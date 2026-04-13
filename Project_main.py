import Project_env
import Project_agent
import os
import time

import numpy as np


def training_agent(learning_file_name, return_file_name, env, num_episodes, eps, alpha, eps_min=-1, n_dec=-1, agent_type="Q-Learning"):
    agent = None
    if agent_type == "Q-Learning":
        agent = Project_agent.QLearner(file_name=learning_file_name, n_dec=n_dec, eps_min=eps_min)
    agent.set_epsilon(eps)
    agent.set_alpha(alpha)
    agent.play_episodes(env, num_episodes)
    agent.save_learning_table()
    np.savetxt(return_file_name, agent.returns, delimiter=',')

    return agent


def run_agent(learning_file_name, return_file_name, env, num_episodes, eps, alpha, agent_type="Q-Learning"):
    agent = None
    if agent_type == "Q-Learning":
        agent = Project_agent.QLearner(file_name=learning_file_name)
    agent.set_epsilon(eps)
    agent.set_alpha(alpha)
    agent.disable()

    if os.path.exists(learning_file_name):
        agent.load_learning_table()
    agent.play_episodes(env, num_episodes)
    np.savetxt(return_file_name, agent.returns, delimiter=',')

    return agent


if __name__ == "__main__":
    before_training = False
    training = True

    learning_file_name = 'q_learner.csv'
    num_episodes = 500_000
    n_dec = 200_000
    eps = 0.7
    eps_min = 0.1
    alpha = 0.1
    env = Project_env.BoardGame(n=4, print_states=False)

    if before_training is True:
        return_file_name = 'Project_experiment_before_training.csv'
        num_episodes = 1000
        start_time = time.time()
        agent = run_agent(learning_file_name,
                          return_file_name,
                          env,
                          num_episodes,
                          eps,
                          alpha,
                          agent_type="Q-Learning")
        finish_time = time.time()
        print(f"Number of more games won than environment: {sum(agent.returns)}")
        print(f"Total time in minutes: {(finish_time - start_time)/60}")

    if training is True:
        return_file_name = 'Project_experiment_train.csv'
        start_time = time.time()
        agent = training_agent(learning_file_name,
                               return_file_name,
                               env,
                               num_episodes,
                               eps,
                               alpha,
                               eps_min=eps_min,
                               n_dec=n_dec,
                               agent_type="Q-Learning")
        finish_time = time.time()
        print(f"Average training returns: {np.mean(agent.returns)}")
        print(f"Number of visited states: {len(agent.q_table.keys())}")
        print(f"Total time in minutes: {(finish_time - start_time)/60}")

    else:
        return_file_name = 'Project_experiment_test.csv'
        num_episodes = 1000
        start_time = time.time()
        agent = run_agent(learning_file_name,
                          return_file_name,
                          env,
                          num_episodes,
                          eps,
                          alpha,
                          agent_type="Q-Learning")
        finish_time = time.time()
        print(f"Number of more games won than environment: {sum(agent.returns)}")
        print(f"Total time in minutes: {(finish_time - start_time)/60}")
