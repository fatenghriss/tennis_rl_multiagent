# tennis_rl_multiagent
This project consists in developing a multi-agent reinforcement learning algorithm to train agents to play table tennis.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The **observation space** consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an **average score of +0.5** (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.
## Installation
First, and in order to have a working environment that is as clean as possible, let's:
1. Create and activate a new environment with Python 3.6
- __Linux__ or __Mac__:
```
conda create --name tennis_rl python=3.6
source activate tennis_rl
```
- __Windows__:
```
conda create --name tennis_rl python=3.6
activate tennis_rl
```
2. Install dependencies:

```
pip install .
```

## Training

In order to train your agent and test your agent, all you have to do is the following:

```
python main.py
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
