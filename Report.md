[//]: # (Image References)

[image1]: agents_training.gif "Trained Agent"
[image2]: reward_history.png "Rewards Plot"

# Project Report

## Task: Reacher
This project demonstrates how to use Deep Reinforcement Learning to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

| ![Trained Agent][image1] |
|:--:| 
| *<p style="font-size: 12px; color: gray;">Agents during training. Lagspikes come from optimizing the agents every 20 steps</p>* |

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

## Solution: Multi-Agent Deep Deterministic Policy Gradient
To solve this environment, both agents used a multi-agent implementation of DDPG:
- **Actor Networks**: The actor is a normal FFNN with 3 hidden layers, and a Tanh activation at the end to restrict the range between -1 and 1. The actor uses 2 networks for learning stability.
- **Critic Networks**: The critic is composed of a FFNN with two separate inputs, one for the full state and another for the actions of all agents. These inputs are at first treated separately, but later concatenated to generate the final value prediction.

##### Multi-agent-related changes
To adapt the agents for a multi-agent approach, the critics have access to the states of all the agents, as well as their actions. Additionally, given that this environment requires collaboration, **both agents share the same critic**.

### Experience Replay
This implementation includes a memory buffer for storing experience. It is then used to sample batches of data to train the networks. Each agent has their own memory buffer.

### Exploration-Exploitation
At first, exploration was done through adding noise to the actions output. Following [OpenAI's Spinning Up explanation](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#exploration-vs-exploitation), gaussian noise was added, and the magnitude of noise decreased during the learning procedure. This process did not yield great results, and learning did not converge. Because of this, I decided to go with an epsilon-greedy method. Noise magnitude is therefore fixed, and only added to the action vector with a probability of `epsilon`. 

Training begins with `epsilon=0.9` and gradually decreases after every training step. Epsilon is capped above `epsilon=0.01`, to ensure a small degree of exploration.

### Results

![Rewards Plot][Image2]
The graph above demonstrates the agent's performance per episode, as well as the 100 Episode Average Reward. As explained on the project instructions, training seems to be very unstable, and requires a lot of time to reach reasonable results. Following the graph above, we see that the environment was solved at around **3250** episodes. This is only temporary, as learning diverged after a few hundred episodes. The training procedure took in total `5000 episodes`, and during that period the agents went from solving the environment to getting bad performance 3 times. This shows how unstable multi-agent training can be, at least in a vanilla configuraiton like the one used here.

#### Hyperparameters
After a lot of tweaking, these are the hyperparameters that achieved good performance:
| Hyperparameter | Value                                   |
|----------------|-----------------------------------------|
| Actor layers   | `[24, 512, 128, 2] `                      |
| Critic layers  | `[48, 256]┓`<br>`[4, 256]━┻━[512, 128, 64, 1]`|
| Activation Function  | ELU                                     |
| Actor Learning Rate  | 0.0001                                  |
| Critic Learning Rate | 0.0001                                  |
| Batch Size           | 256                                     |
| Update Rate          | 20                                      |
| Discount Factor      | 0.995                                   |
| Alpha                | 0.0003                                  |
| Epsilon              | 0.9                                     |
| Epsilon Decay        | 0.99995                                 |
| Min Epsilon          | 0.01                                    |
| Memory Buffer Size   | 16,348                                  |

### Next Steps

This project was much harder than previous ones, and it raised many questions regarding the current implementation and possible future directions. My main concern is: **How can learning be more stable?**

Here are a few ideas I'd like to explore:

- **Self-play**: Both agents are solving a pretty much identical problem, and because of that, their experience and "point of view" of the problem can be shared across agents. Self-play was explored by sharing a memory buffer, and mirroring the state and action vectors. This didn't yield good results, although it could have been because of the hyperparameters chosen at the time, and not because of self-play itself.
- **Prioritized Experience Replay**: This problem is heavily skewed towards experience with zero reward. This makes it very hard for the critic to propagate expected reward because of the sparsity of rewards. Using prioritized experience replay could aid at stabilizing and accelerating learning. Additionally, a few ideas have come up. For example: **Could there be a separate priority for learning between the critic and the actor?** Say by sampling different experience for the critic vs actor.
- **Explore hyperparameters**: Because of how long it takes for the agents to learn, a thorough exploration of hyperparameters was not achieved. It might be that the current hyperparameters worked out of luck. During my small hyperparameter optimization, I notice performance increase by inducing small changes to parameters like `discount`, `epsilon` and `update_rate`. Is this solution really sensible to those parameters? Or was that performance increase due to stochasticity in the environment initial conditions?
- **Explore agent-communication**: Although this problem doesn't require communication between agents, it'd be interesting to learn how this kind of communication is implemented and handled for multi-agent environments.
- **Solve the soccer environment**: Lastly, once stable learning is achieved, it'd be interesting to see if such a configuration can be used for the `Soccer` environment with little modifications.
