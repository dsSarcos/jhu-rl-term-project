from Project_env import BoardGame
from Project_agent import RLAgent, QLearner, eSARSA

from datetime import datetime
import os
import numpy as np
from pathlib import Path

from multiprocessing import Pool
from tqdm import tqdm

import random
random.seed(100)

experiment_dir = "experiments"

def test(agent1 = None, agent2=None, n_episodes=1000, in_train=True):
    if not agent1:
        agent1 = RLAgent(eps=1)
    else:
        agent1.disable()
    if not agent2:
        agent2 = RLAgent(eps=1)
    else:
        agent2.disable()
    p1_results = np.zeros(n_episodes)
    environment = BoardGame()
    for i in range(n_episodes):
        reward = 0
        game_end = False
        current_turn = False
        while not game_end:
            roll = random.choices([0, 1, 2, 3, 4], weights=[1/16, 1/4, 3/8, 1/4, 1/16]).pop()
            current_state = (environment.p1_states, environment.p2_states) if not current_turn else (environment.p2_states, environment.p1_states)
            actions_indices = environment.get_actions(*current_state, roll)
            if actions_indices:
                 action = agent1.select_action(environment.encode_state(*current_state), actions_indices, current_turn) if not current_turn else agent2.select_action(environment.encode_state(*current_state), actions_indices, current_turn)
                 next_state, reward, game_end, current_turn = environment.execute_action(current_state, action, roll, current_turn)
                 current_state = next_state
            else:
                current_turn = not current_turn

        p1_results[i] = reward if current_turn else -reward
        environment.reset()

    p2_results = np.zeros(n_episodes)
    environment = BoardGame()
    for i in range(n_episodes):
        reward = 0
        game_end = False
        current_turn = False
        while not game_end:
            roll = random.choices([0, 1, 2, 3, 4], weights=[1/16, 1/4, 3/8, 1/4, 1/16]).pop()
            current_state = (environment.p1_states, environment.p2_states) if not current_turn else (environment.p2_states, environment.p1_states)
            actions_indices = environment.get_actions(*current_state, roll)
            if actions_indices:
                 action = agent2.select_action(environment.encode_state(*current_state), actions_indices, current_turn) if not current_turn else agent1.select_action(environment.encode_state(*current_state), actions_indices, current_turn)
                 next_state, reward, game_end, current_turn = environment.execute_action(current_state, action, roll, current_turn)
                 current_state = next_state
            else:
                current_turn = not current_turn

        p2_results[i] = reward if current_turn else -reward
        environment.reset()

    agent1.enable()
    agent2.enable()
    if in_train:
        return len(p1_results[p1_results == 1]) / len(p1_results), len(p2_results[p2_results == -1]) / len(p2_results)
    else:
        return np.asarray([p1_results, p2_results]).T


def train():
    experiment_name = "training"
    n_agents = 1

    num_episodes = 400_000

    q_learner_hyperparameters = {
        "n_dec": 250_000,
        "eps": 0.9,
        "eps_min": 0.1,
        "alpha": 0.1
    }

    onp_esarsa_hyperparameters = {
        "n_dec": 250_000,
        "eps": 0.9,
        "eps_min": 0.1,
        "alpha": 0.1,
        "on_policy": True
    }

    ofp_esarsa_hyperparameters = {
        "n_dec": 250_000,
        "eps": 0.9,
        "eps_min": 0.1,
        "alpha": 0.1,
        "on_policy": False
    }


    q_returns = np.zeros((n_agents, num_episodes, 2))
    q_curve = []

    onp_esarsa_returns = np.zeros((n_agents, num_episodes, 2))
    onp_es_curve = []

    ofp_esarsa_returns = np.zeros((n_agents, num_episodes, 2))
    ofp_es_curve = []

    ql, onp_es, ofp_es = None, None, None
    environment = BoardGame()
    for i in range(n_agents):
        ql = QLearner(**q_learner_hyperparameters)
        onp_es = eSARSA(**onp_esarsa_hyperparameters)
        ofp_es = eSARSA(**ofp_esarsa_hyperparameters)
        for j in tqdm(range(num_episodes)):
            if j % 5000 == 0:
                q_curve.append(test(ql))
                onp_es_curve.append(test(onp_es))
                ofp_es_curve.append(test(ofp_es))
            q_returns[i, j, 0], q_returns[i, j, 1] = ql.play_episodes(environment, j)
            onp_esarsa_returns[i, j, 0], onp_esarsa_returns[i, j, 1] = onp_es.play_episodes(environment, j)
            ofp_esarsa_returns[i, j, 0], ofp_esarsa_returns[i, j, 1] = ofp_es.play_episodes(environment, j)
            
    fp = Path(f"{experiment_dir}/{experiment_name}/agents/q_learning.parquet")
    fp.parent.mkdir(parents=True, exist_ok=True)
    ql.save_learning_table(fp)

    fp = Path(f"{experiment_dir}/{experiment_name}/agents/onp_eSARSA.parquet")
    fp.parent.mkdir(parents=True, exist_ok=True)
    onp_es.save_learning_table(fp)

    fp = Path(f"{experiment_dir}/{experiment_name}/agents/ofp_eSARSA.parquet")
    fp.parent.mkdir(parents=True, exist_ok=True)
    ofp_es.save_learning_table(fp)

    np.save(Path(f"{experiment_dir}/{experiment_name}/ql_test_curve.npy"), np.array(q_curve))
    np.save(Path(f"{experiment_dir}/{experiment_name}/onp_es_test_curve.npy"), np.array(onp_es_curve))
    np.save(Path(f"{experiment_dir}/{experiment_name}/ofp_es_test_curve.npy"), np.array(ofp_es_curve))

    np.save(Path(f"{experiment_dir}/{experiment_name}/ql_history.npy"), q_returns)
    np.save(Path(f"{experiment_dir}/{experiment_name}/onp_es_history.npy"), onp_esarsa_returns)
    np.save(Path(f"{experiment_dir}/{experiment_name}/ofp_es_history.npy"), ofp_esarsa_returns)

    return ql, onp_es, ofp_es

if __name__ == "__main__":
    ql, onp_es, ofp_es = None, None, None
    
    # Training: H1, H2
    ql, onp_es, ofp_es = train()

    if not (ql and onp_es and ofp_es):
        ql = RLAgent(eps=0)
        onp_es = RLAgent(eps=0)
        ofp_es = RLAgent(eps=0)

        ql.load_learning_table(Path("experiments/training/agents/q_learning.parquet"))
        onp_es.load_learning_table(Path("experiments/training/agents/onp_eSARSA.parquet"))
        ofp_es.load_learning_table(Path("experiments/training/agents/ofp_eSARSA.parquet"))

    # Benchmark: Random vs Random, 1000 games each
    np.savetxt(Path("experiments/random_v_random.csv"), test(in_train=False), delimiter=",")

    # H3
    np.savetxt(Path("experiments/ql_v_onp.csv"), test(ql, onp_es, in_train=False), delimiter=",")
    np.savetxt(Path("experiments/ql_v_ofp.csv"), test(ql, ofp_es, in_train=False), delimiter=",")
    np.savetxt(Path("experiments/onp_v_ofp.csv"), test(onp_es, ofp_es, in_train=False), delimiter=",")