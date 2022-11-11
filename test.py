from agents.maddpg import MADDPG
from agents.ddpg import MemoryBuffer, ActorFeedForward, CriticFeedForward

from unityagents import UnityEnvironment
import numpy as np

def test(env, agents, num_runs=5):
    # get the default brain
    brain_name = env.brain_names[0]

    for run_idx in range(1, num_runs+1):
        env_info = env.reset(train_mode=False)[brain_name]
        epoch_cum_rewards = None
        while True:
            states = env_info.vector_observations
            actions = []
            for state, agent in zip(states, agents):
                action = agent.act(state).detach().numpy()
                actions.append(action)
            actions = np.array(actions)
            env_info = env.step(actions)[brain_name]
            rewards = np.array(env_info.rewards)
            dones = env_info.local_done

            if epoch_cum_rewards is None:
                epoch_cum_rewards = np.array(rewards)
            else:
                epoch_cum_rewards += np.array(rewards)

            if np.any(dones):
                break
        epoch_reward = np.mean(epoch_cum_rewards)
        print(f"Run {run_idx} avg performance: {epoch_reward}")

if __name__ == '__main__':
    actor1 = ActorFeedForward([24, 512, 128, 2])
    actor2 = ActorFeedForward([24, 512, 128, 2])
    critic1 = CriticFeedForward([24 * 2, 256], [2 * 2, 256], [512, 128, 64, 1])
    critic2 = CriticFeedForward([24 * 2, 256], [2 * 2, 256], [512, 128, 64, 1])
    # Self-play is incorporated by sharing the memory buffer between agents
    agent1 = MADDPG(actor1, critic1, MemoryBuffer())
    agent2 = MADDPG(actor2, critic1, MemoryBuffer())
    env = UnityEnvironment(file_name='Tennis.app')
    agent1.load("agent_checkpoints_0")
    agent2.load("agent_checkpoints_1")

    test(env, MADDPG.agents)