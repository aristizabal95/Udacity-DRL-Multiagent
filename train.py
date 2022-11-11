from agents.ddpg import MemoryBuffer, ActorFeedForward, CriticFeedForward
from agents.maddpg import MADDPG

from unityagents import UnityEnvironment
from tqdm import tqdm
import numpy as np
import json

def move_front(arr, idx):
    """Moves an element at position idx to the first position in the array.

    Args:
        arr (np.array): Numpy array 
        idx (int): Index of element to move to the front
    """
    el = arr[[idx]]
    other_els = np.delete(arr, idx, 0)
    sorted_arr = np.concatenate((el, other_els))
    return sorted_arr

def train(env, agents, num_epochs=5000, epsilon=0.1, epsilon_decay=1e-6, min_epsilon=0.01):
    # get the default brain
    brain_name = env.brain_names[0]
    loop = tqdm(range(num_epochs))
    best_reward = float("-inf")
    rewards_history = []

    for epoch in loop:
        showcase = not epoch % 25 == 0
        env_info = env.reset(train_mode=showcase)[brain_name]
        epoch_cum_rewards = None
        while True:
            states = env_info.vector_observations
            actions = []
            for state, agent in zip(states, agents):
                action = agent.act(state, epsilon=epsilon)
                actions.append(action.detach().numpy())
            actions = np.array(actions)
            env_info = env.step(actions)[brain_name]
            rewards = np.array(env_info.rewards)
            next_states = env_info.vector_observations
            dones = np.array(env_info.local_done).astype(np.float32)
            for i, agent in enumerate(agents):
                # order experience in the perspective of the current agent
                exp = (states, actions, rewards, next_states, dones)
                agent.step(*exp)

            epsilon = epsilon * (1 - epsilon_decay)
            epsilon = max(epsilon, min_epsilon)

            if epoch_cum_rewards is None:
                epoch_cum_rewards = np.array(rewards)
            else:
                epoch_cum_rewards += np.array(rewards)

            if np.any(dones):
                break
        epoch_reward = np.mean(epoch_cum_rewards)
        if epoch_reward > best_reward:
            best_reward = epoch_reward
            for i, agent in enumerate(agents):
                agent.save(f"agent_checkpoints_{i}".replace(".", "_"))
        rewards_history.append(epoch_reward)
        loop.set_description(f"Avg Reward: {round(np.mean(rewards_history[-100:]), 4)} | Epsilon: {round(epsilon, 3)}")

    with open("reward_history.json", "w") as f:
        json.dump(rewards_history, f)


if __name__ == '__main__':
    actor1 = ActorFeedForward([24, 512, 128, 2])
    actor2 = ActorFeedForward([24, 512, 128, 2])
    critic1 = CriticFeedForward([24 * 2, 256], [2 * 2, 256], [512, 128, 64, 1])
    critic2 = CriticFeedForward([24 * 2, 256], [2 * 2, 256], [512, 128, 64, 1])
    # Self-play is incorporated by sharing the memory buffer between agents
    agent1 = MADDPG(actor1, critic1, MemoryBuffer())
    agent2 = MADDPG(actor2, critic1, MemoryBuffer())
    env = UnityEnvironment(file_name='Tennis.app')

    train(env, MADDPG.agents, epsilon=.9, epsilon_decay=5e-5)
