import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np


"""
Chain Environment from Bootstrapped DQN paper.
"""
class ChainBstrapDQN(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
            n=10,
            lrwd=1e-3,
            rrwd=1,
            start_state=2,
            state_enc='therm',
            length_factor=5):

        super(ChainBstrapDQN).__init__()

        # Chain length
        self.n = n 
        assert n > 3, "Chain length has to be more than 3"
        self.lrwd = lrwd 
        self.rrwd = rrwd

        assert state_enc in ['therm', 'one-hot'], "Only thermnal and one-hot encoding supported" 
        self.state_enc = state_enc

        # This is from the bootstrapped DQN paper.
        self.episode_len = n + 10

        self.start_state = start_state
        self.state = start_state 

        # Max return is 1, episode terminates when RightMost state is reached.
        self.max_return =  max(self.rrwd, self.episode_len * self.lrwd)

        # 0 : Left, 1 : Right
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=np.float32)

        self.done = False
        
        self.steps = 0

    def step(self, action):

        self.steps += 1

        if self.state == 1:
            reward = self.lrwd if action == 0 else 0
        elif self.state == self.n:
            reward = self.rrwd if action == 1 else 0
        else:
            reward = 0

        self.done = ((self.steps == (self.episode_len - 1) ) or self.state == self.n)

        if action == 0:
            # LEFT action
            self.state = max(1, self.state - 1)
        else:
            # RIGHT action
            self.state = min(self.n, self.state + 1)
        return self._enc_state(self.state), reward, self.done, {} 

    def reset(self):

        self.steps = 0 
        self.state = self.start_state
        self.done = False

        return self._enc_state(self.state)

    def _enc_state(self, state):
        """
        Encode the state to generate observation.
        """
        obs = self.observation_space.sample()
        obs[:] = 0

        if self.state_enc == 'one-hot':
            # One-Hot Encoding
            obs[state] = 1
        else:
            # Thermal Encoding
            r = np.arange(self.n)
            obs[r[r < state]] = 1
        
        return obs

    def _encode_all_states(self):
        _enc = np.vstack([self._enc_state(s) for s in range(1, self.n + 1)])

        return _enc


    def render(self, mode='human'):
        pass
    def close(self):
        pass

