import numpy as np
import random

class RandomAgent:

    def __init__(self, num_agents, state_size, action_size, seed):
        """Initialize an Agent obje

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

    def act(self, state, eps=0.):
        """Choose a random action."""
        actions = np.random.randn(num_agents, action_size)
        return np.clip(actions, -1, 1)

    def learn(self):
        """No learning."""
        pass
