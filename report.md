# Report about solving the Tennis environment

In this report, we will discuss the following:
* The learning algorithm
* The network architecture
* The reward plot

## Learning algorithm
In order to solve the tennis environment, we implemented the MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm.

MADDPG is a model-free off-policy actor critic algorithm, that basically combines DQN and DPG. What DDPG brings to the table is being able to solve continuous actions spaces and to learn a deterministic policy, while using the actor-critic framework.
This policy-gradient algorithm uses two deep neural networks to learn the best policy for reward maximization and explore a stochastic environment.
DDPG is well-known for solving continuous spaces and the multi-agent here is used to improve coordination.

Basically, the Actor and Critic of each agent will work together to converge more quickly.

We also applied noise for an exploration policy and used the replay buffer for a better memorization.

### Hyperparameters:
The following hyper-parameters have been used to train the model. A few have been inspired from the paper and others such as the batch size and learning rates have been adjusted with experiments:
* BATCH_SIZE = 128        # minibatch size
* GAMMA = 0.99            # discount factor
* TAU = 6e-2              # for soft update of target parameters
* LR_ACTOR = 1e-3         # learning rate of the actor
* LR_CRITIC = 1e-3        # learning rate of the critic
* WEIGHT_DECAY = 0        # L2 weight decay
* UPDATE_EVERY = 1        # time steps between network updates
* N_UPDATES = 1           # number of times training
* eps_start = 6           # Noise level start
* eps_end = 0             # Noise level end
* eps_decay = 250         # Number of episodes to decay over from start to end


### Actor Network architecture:
For the actor network architecture, we decided to go with the following:
* First fully connected layer with input's size = state space size*2 and output's size = 256
* Second fully connected layer with input's size = 256 and the output's size = 128
* Third fully connected network with input's size = 128 and the output's size is the action's size

### Critic Network Architecture:
For the critic network architecture, we decided to go with the following:
* First fully connected layer with input's size = state space size*2 and output's size = 256
* Second fully connected layer with input's size = 256 +(actions size * 2) and the output's size = 128
* Third fully connected network with input's size = 128 and the output's size = 1 to map states and actions to Q-values

## Reward plot:
It took 834 episodes to DDPG to solve the environment.
The following is the plot of the rewards

![Rewards per episode](reward_plot.png)

For more information, please refer to the following files:
* [model.py](training/model.py) to get the code for the network architecture
* [ddpg.py](agents/ddpg.py) to get the code for the agent's implementation
* [train.py](train.py) to get the code for training

## Improvement
I believe that I would be interested to verify if [SAC](https://arxiv.org/abs/1801.01290) (Soft Actor Critic) would give better results, due to the fact that it incorporates the entropy measure of the policy into the reward to encourage exploration. We can also explore attention mechanisms that can improve the communication between agents.