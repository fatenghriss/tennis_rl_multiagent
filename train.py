from unityagents import UnityEnvironment
import numpy as np
import random
from agents.ddpg import DDPGAgent
import torch
from training.train import train
import matplotlib.pyplot as plt

def main():

    env = UnityEnvironment(file_name="/home/faten/projects/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis.x86")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations
    state = np.reshape(state, (1, 48))
    state_size = len(state)
    print("creating agents")
    agent0 = DDPGAgent(state_size, action_size, 2, 0)
    agent1 = DDPGAgent(state_size, action_size, 2, 0)
    print("agents created")
    scores = train(env, agent0, agent1)

    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Epsiode #')
    plt.show()

    state = env.reset()[brain_name]
    for t in range(200):
        action = agent.act(state, add_noise=False)
        env.render()
        state, reward, done, _ = env.step(action)[brain_name]
        if done:
            break

    env.close()

main()