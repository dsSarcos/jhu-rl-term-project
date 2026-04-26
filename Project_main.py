from Project_env import BoardGame
from Project_agent import RLAgent, QLearner, eSARSA

from datetime import datetime
import os
import numpy as np
from pathlib import Path

from multiprocessing import Pool
from tqdm import tqdm

import random
random.seed(1)

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
    n_agents = 20

    num_episodes = 400_000
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
        "alpha": 0.1
    }

    # q_learners = [QLearner(**q_learner_hyperparameters) for _ in range(n_agents)]
    q_returns = np.zeros((n_agents, num_episodes, 2))

    # esarsa_learners = [eSARSA(**esarsa_hyperparameters) for _ in range(n_agents)]
    esarsa_returns = np.zeros((n_agents, num_episodes, 2))


    environment = BoardGame()
    for i in range(n_agents):
        print(f"Agent: {i+1}")
        ql = QLearner(**q_learner_hyperparameters)
        es = eSARSA(**esarsa_hyperparameters)
        for j in tqdm(range(num_episodes)):
            q_returns[i, j, 0], q_returns[i, j, 1] = ql.play_episodes(environment, j)
            esarsa_returns[i, j] = es.play_episodes(environment, j)
            
        if i == 0:
            # fp = Path(f"{experiment_dir}/{experiment_name}/agents/q_learning/{i}.csv")
            fp = Path(f"{experiment_dir}/{experiment_name}/agents/q_learning/{i}.parquet")
            fp.parent.mkdir(parents=True, exist_ok=True)
            ql.save_learning_table(fp)

            # fp = Path(f"{experiment_dir}/{experiment_name}/agents/eSARSA/{i}.csv")
            fp = Path(f"{experiment_dir}/{experiment_name}/agents/eSARSA/{i}.parquet")
            fp.parent.mkdir(parents=True, exist_ok=True)
            es.save_learning_table(fp)
        
        environment.reset()

    data = np.array([q_returns[:, :, 0].mean(axis=0), esarsa_returns[: , :, 0].mean(axis=0)]).T
    np.savetxt(Path(f"{experiment_dir}/{experiment_name}/player1_returns.csv"), data, delimiter=",")
    data = np.array([q_returns[:, :, 1].mean(axis=0), esarsa_returns[: , :, 1].mean(axis=0)]).T
    np.savetxt(Path(f"{experiment_dir}/{experiment_name}/player2_returns.csv"), data, delimiter=",")

    np.save(Path(f"{experiment_dir}/{experiment_name}/ql_history.npy"), q_returns)
    np.save(Path(f"{experiment_dir}/{experiment_name}/es_history.npy"), esarsa_returns)

@timer
def experiment_2():
    n_games = 10_000

    player1 = RLAgent(eps=0)
    player2 = RLAgent(eps=0)

    player_rand = RLAgent(eps=1)

    player1.load_learning_table(Path(f"{experiment_dir}/{1}/agents/q_learning/{0}.csv"))
    print("Player 1 loaded")
    player2.load_learning_table(Path(f"{experiment_dir}/{1}/agents/eSARSA/{0}.csv"))
    print("Player 2 loaded")

    ql_results = np.zeros(n_games)
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
                 action = player1.select_action(environment.encode_state(*current_state), actions_indices, current_turn) if not current_turn else player_rand.select_action(environment.encode_state(*current_state), actions_indices, current_turn)
                 next_state, reward, game_end, current_turn = environment.execute_action(current_state, action, roll, current_turn)
                 current_state = next_state
            else:
                current_turn = not current_turn

        ql_results[i] = reward if current_turn else -reward
        environment.reset()


    es_results = np.zeros(n_games)
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
                 action = player2.select_action(environment.encode_state(*current_state), actions_indices, current_turn) if not current_turn else player_rand.select_action(environment.encode_state(*current_state), actions_indices, current_turn)
                 next_state, reward, game_end, current_turn = environment.execute_action(current_state, action, roll, current_turn)
                 current_state = next_state
            else:
                current_turn = not current_turn

        es_results[i] = reward if current_turn else -reward
        environment.reset()


    print(f"Player 1 WR: {len(ql_results[ql_results == 1]) / len(ql_results)}")
    print(f"Player 2 WR: {len(es_results[es_results == -1]) / len(es_results)}")

@timer
def experiment_3():
    n_games = 10_000

    player1 = RLAgent(eps=0)
    player2 = RLAgent(eps=0)

    player2.load_learning_table(Path(f"{experiment_dir}/{1}/agents/q_learning/{0}.csv"))
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
                 action = player1.select_action(environment.encode_state(*current_state), actions_indices, current_turn) if not current_turn else player2.select_action(environment.encode_state(*current_state), actions_indices, current_turn)
                 next_state, reward, game_end, current_turn = environment.execute_action(current_state, action, roll, current_turn)
                 current_state = next_state
            else:
                current_turn = not current_turn

        results[i] = reward if current_turn else -reward
        environment.reset()

        

if __name__ == "__main__":
    
    # Training
    experiment_1()
    
    # Q_learning vs random, eSARSA vs random (as player one and player two, 4 games)
    # experiment_2()

    # Q_learning vs eSARSA (as player one and player two, 2 games)
    # experiment_3()