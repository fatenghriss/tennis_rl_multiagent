# Report about solving the Tennis environment

In this report, we will discuss the following:
* The learning algorithm
* The network architecture
* The reward plot

## Learning algorithm
In order to solve the tennis environment, we implemented the DDPG (Deep Deterministic Policy Gradient) algorithm.

### Hyperparameters:
The following hyperparameters have been used to train the model:
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
