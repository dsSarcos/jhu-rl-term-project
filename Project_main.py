import Project_env
import Project_agent

from datetime import datetime
import os
import numpy as np
from pathlib import Path

from multiprocessing import Pool

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

def timer(f):
    def g():
        start_time = datetime.now()
        f()
        end_time = datetime.now()
        print(str(end_time - start_time))
    return g

experiment_dir = "experiments"

@timer
def experiment_1():
    experiment_name = "1"
    n_agents = 1

    num_episodes = 250_000
    n_dec = 250_000
    eps = 0.9
    eps_min = 0.1

    q_learner_hyperparameters = {
        "n_dec": n_dec,
        "eps": eps,
        "eps_min": eps_min,
        "alpha": 0.1
    }

    esarsa_hyperparameters = {
        "n_dec": n_dec,
        "eps": eps,
        "eps_min": eps_min,
        "alpha": 1.0
    }

    q_learners = [Project_agent.QLearner(**q_learner_hyperparameters) for _ in range(n_agents)]
    q_returns = np.zeros((n_agents, num_episodes))

    esarsa_learners = [Project_agent.eSARSA(**esarsa_hyperparameters) for _ in range(n_agents)]
    esarsa_returns = np.zeros((n_agents, num_episodes))


    environment = Project_env.BoardGame()
    for j in range(num_episodes):
        if j % 1000 == 0:
            print(f"Episode: {j}")
        for i, (ql, es) in enumerate(zip(q_learners, esarsa_learners)):
            q_returns[i, j] = ql.play_episodes(environment, j)
            esarsa_returns[i, j] = es.play_episodes(environment, j)
            
    for i, (ql, es) in enumerate(zip(q_learners, esarsa_learners)):
        fp = Path(f"{experiment_dir}/{experiment_name}/agents/q_learning/{i}.csv")
        fp.parent.mkdir(parents=True, exist_ok=True)
        ql.save_learning_table(fp)

        fp = Path(f"{experiment_dir}/{experiment_name}/agents/eSARSA/{i}.csv")
        fp.parent.mkdir(parents=True, exist_ok=True)
        es.save_learning_table(fp)

    data = np.array([q_returns.mean(axis=0), esarsa_returns.mean(axis=0)]).T
    np.savetxt(Path(f"{experiment_dir}/{experiment_name}/returns.csv"), data, delimiter=",")


if __name__ == "__main__":
    
    experiment_1()