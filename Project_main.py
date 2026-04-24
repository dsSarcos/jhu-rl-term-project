from Project_env import BoardGame
from Project_agent import RLAgent, QLearner, eSARSA

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
        agent = QLearner(file_name=learning_file_name, n_dec=n_dec, eps_min=eps_min)
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
        agent = QLearner(file_name=learning_file_name)
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

    num_episodes = 10
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

    q_learners = [QLearner(**q_learner_hyperparameters) for _ in range(n_agents)]
    q_returns = np.zeros((n_agents, num_episodes))

    esarsa_learners = [eSARSA(**esarsa_hyperparameters) for _ in range(n_agents)]
    esarsa_returns = np.zeros((n_agents, num_episodes))


    environment = BoardGame()
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


@timer
def experiment_2():
    n_games = 100_000

    player1 = RLAgent(eps=0)
    player2 = RLAgent(eps=0)

    player1.load_learning_table(Path(f"{experiment_dir}/{1}/agents/q_learning/{0}.csv"))
    print("Player 1 loaded")
    player2.load_learning_table(Path(f"{experiment_dir}/{1}/agents/eSARSA/{0}.csv"))
    print("Player 2 loaded")

    results = np.zeros(n_games)
    environment = BoardGame()
    
    for i in range(n_games):
        if i % 1000 == 0:
            print(f"Game: {i}")
        reward = 0

        game_end = False
        current_turn = False
        while not game_end:
            roll = random.choices([0, 1, 2, 3, 4], weights=[1/16, 1/4, 3/8, 1/4, 1/16]).pop()
            current_state = (environment.p1_states, environment.p2_states) if not current_turn else (environment.p2_states, environment.p1_states)
            actions_indices = environment.get_actions(*current_state, roll)
            if actions_indices:
                 action = player1.select_action(environment.encode_state(*current_state), actions_indices) if not current_turn else player2.select_action(environment.encode_state(*current_state), actions_indices)
                 next_state_and_turn = environment.transition(*current_state, action, roll, current_turn)
                 next_state, current_turn = (next_state_and_turn[:2], next_state_and_turn[-1])
                 current_state = next_state
            else:
                current_turn = not current_turn

            game_end = environment.get_terminal_flag(*current_state)

        results[i] = 1 if current_turn else -1

    print(f"Player 1 WR: {len(results[results == 1]) / len(results)}")
    print(f"Player 2 WR: {len(results[results == -1]) / len(results)}")



        

if __name__ == "__main__":
    
    experiment_1()
    
    experiment_2()