import Project_env
import Project_agent

from datetime import datetime
import os
import numpy as np

import random
random.seed(1)


def training_agent(learning_file_name,
                   return_file_name,
                   env,
                   num_episodes,
                   eps,
                   alpha,
                   eps_min=-1.0,
                   n_dec=-1.0,
                   agent_type="Q-Learning",
                   retrain=False):
    agent = None
    if agent_type == "Q-Learning":
        agent = Project_agent.QLearner(file_name=learning_file_name, n_dec=n_dec, eps_min=eps_min)
    agent.set_epsilon(eps)
    agent.set_alpha(alpha)
    if retrain is True:
        agent.load_learning_table()
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

    if learning_file_name and os.path.exists(learning_file_name):
        agent.load_learning_table()
    agent.play_episodes(env, num_episodes)
    np.savetxt(return_file_name, agent.returns, delimiter=',')

    return agent



if __name__ == "__main__":
    before_training = False
    training = True
    testing = False

    num_episodes = 600_000
    n_dec = 250_000
    eps = 0.8
    eps_min = 0.1
    alpha = 0.1
    env = Project_env.BoardGame(n=4, print_states=False)

    if before_training is True:
        return_file_name = 'Project_experiment_before_training.csv'
        learning_file_name = None
        num_episodes = 1000
        start_time = datetime.now()
        agent = run_agent(learning_file_name,
                          return_file_name,
                          env,
                          num_episodes,
                          eps,
                          alpha,
                          agent_type="Q-Learning")
        finish_time = datetime.now()
        print(f"Number of more games won than environment: {sum(agent.returns)}")
        print(f"Total time: {str(finish_time - start_time)}")

    elif training is True:
        return_file_name = 'Project_experiment_speed.csv'
        learning_file_name = 'q_learner.csv'
        start_time = datetime.now()
        agent = training_agent(learning_file_name,
                               return_file_name,
                               env,
                               num_episodes,
                               eps,
                               alpha,
                               eps_min=eps_min,
                               n_dec=n_dec,
                               agent_type="Q-Learning",
                               retrain=False)
        finish_time = datetime.now()
        print(f"Average training returns: {np.mean(agent.returns)}")
        print(f"Number of visited states: {len(agent.q_table.keys())}")
        print(f"Total time: {str(finish_time - start_time)}")

    elif testing is True:
        return_file_name = 'Project_experiment_test.csv'
        learning_file_name = None
        num_episodes = 1000
        start_time = datetime.now()
        agent = run_agent(learning_file_name,
                          return_file_name,
                          env,
                          num_episodes,
                          eps,
                          alpha,
                          agent_type="Q-Learning")
        finish_time = datetime.now()
        print(f"Number of more games won than environment: {sum(agent.returns)}")
        print(f"Total time: {str(finish_time - start_time)}")
