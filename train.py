from unityagents import UnityEnvironment
import numpy as np
import random
from agents.ddpg import DDPGAgent
import torch
import matplotlib.pyplot as plt
from collections import deque


# def main():

env = UnityEnvironment(file_name="/home/faten/projects/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux/Reacher.x86")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

action_size = brain.vector_action_space_size

env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)

n_episodes = 1000

avg_score = []
scores_deque = deque(maxlen=100)
scores = np.zeros(num_agents)
time_steps = 20
update = 10

states = env_info.vector_observations
state_size = states.shape[1]
print("creating agents")
agent0 = DDPGAgent(state_size, action_size, 1, seed=0)
agent1 = DDPGAgent(state_size, action_size, 1, seed=0)
print("agents created")
brain_name = env.brain_names[0]

all_scores = []  # list containing score from each episode
scores_window = deque(maxlen=100)  # last 100 scores
for i_episode in range(1, n_episodes + 1):
    #     print("episode number ", i_episode)
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations
    state = np.reshape(state, (1, 48))
    agent0.reset()
    agent1.reset()
    scores = np.zeros(num_agents)
    while True:
        action0 = agent0.act(state, True)
        action1 = agent1.act(state, True)
        actions = np.concatenate((action0, action1), axis=0)
        actions = np.reshape(actions, (1, 4))
        env_info = env.step(actions)[brain_name]
        next_state = env_info.vector_observations  # get the next state
        next_state = np.reshape(next_state, (1, 48))
        reward = env_info.rewards  # get the reward
        done = env_info.local_done
        agent0.step(state, actions, reward[0], next_state, done, 0)
        agent1.step(state, actions, reward[1], next_state, done, 1)
        ## above step decides whether we will train(learn) the network
        ## actor (local_qnetwork) or we will fill the replay buffer
        ## if len replay buffer is equal to the batch size then we will
        ## train the network or otherwise we will add experience tuple in our
        ## replay buffer.
        state = next_state
        scores += reward
        if np.any(done):
            break
    scores_window.append(np.max(scores))  ## save the most recent score
    all_scores.append(np.max(scores))  ## sae the most recent score

    if i_episode % 10 == 0:
        print('Episode {}\tMax Reward: {:.3f}\tAverage Reward: {:.3f}'.format(
            i_episode, np.max(scores), np.mean(scores_window)))
    if np.mean(scores_window) >= 0.5:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode - 100,
                                                                                     np.mean(scores_window)))
        torch.save(agent0.actor_local.state_dict(), 'checkpoint_actor1.pth')
        torch.save(agent1.actor_local.state_dict(), 'checkpoint_actor2.pth')
        torch.save(agent0.critic_local.state_dict(), 'checkpoint_critic1.pth')
        torch.save(agent1.critic_local.state_dict(), 'checkpoint_critic2.pth')
        break

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(all_scores)+1), all_scores)
plt.ylabel('Score')
plt.xlabel('Epsiode #')
plt.show()

env.close()
