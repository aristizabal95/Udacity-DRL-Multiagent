import torch

from .ddpg import DDPG

def flip(x, dim=0):
    dim_size = x.shape[dim]
    idxs = torch.tensor(range(dim_size)[::-1])
    return x.index_select(dim, idxs)

class MADDPG(DDPG):
    agents = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agents.append(self)
        self.id = len(self.agents) - 1

    def train_critic(self, experience_batch):
        # Decompose the experience into tensors
        states, actions, rewards, next_states, dones = experience_batch
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        # Train the critic with the TD Estimate
        flat_states = states.view(self.batch_size, -1)
        flat_actions = actions.view(self.batch_size, -1)
        rewards = rewards[:, self.id].float() # We only care about the rewards of current agent
        flat_next_states = next_states.view(self.batch_size, -1)
        dones = dones[:, self.id] # We only care about the dones of current agent
        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate(self.agents):
                agent_next_actions = agent.actor_target(next_states[:, i])
                next_actions.append(agent_next_actions)
            next_actions = torch.cat(next_actions, dim=-1)
            q_next = self.critic_target(flat_next_states, next_actions)
        value_estimate = rewards + (1 - dones) * self.discount * q_next
        value_pred = self.critic_local(flat_states, flat_actions)
        critic_loss = self.loss_fn(value_pred, value_estimate)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def train_actor(self, experience_batch):
        # Train the actor by maximizing the critics estimate
        states, actions, *_ = experience_batch

        pred_actions = self.actor_local(states[:, self.id])
        actions[:, self.id] = pred_actions
        flat_states = states.view(self.batch_size, -1)
        flat_actions = actions.view(self.batch_size, -1)
        actor_loss = -self.critic_target(flat_states, flat_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return