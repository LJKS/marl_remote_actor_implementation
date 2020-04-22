import random
import numpy as np
import gym
from scipy.special import softmax

#gyms should implement initialization with None opponent
class Bet_gym():
    def __init__(self, opponent_actor):
        self.turn = 0
        self.finished = False
        self.min_turns = 10
        self.end_prob = 0.05
        self.game_size = 9
        self.opponent = opponent_actor
        self.opponent_state = np.zeros(self.game_size)
        self.player_state = np.zeros(self.game_size)

    def step(self, player_action):
        assert not self.finished
        self.turn = self.turn + 1
        opponent_action, _ = self.opponent.act(self.get_opponent_observation())
        self.opponent_state = self.opponent_state + softmax(np.squeeze(opponent_action))
        self.player_state = self.player_state + softmax(np.squeeze(player_action))
        is_final_step = False
        if self.turn > self.min_turns:
            is_final_step = self.end_prob > np.random.rand()
        if is_final_step:
            self.finished = True
            return self.get_player_observation(), self.agent_won(), True, None
        return self.get_player_observation(), 0, False, None

    def reset(self):
        self.turn = 0
        self.finished = False
        self.opponent_state = np.zeros(self.game_size)
        self.player_state = np.zeros(self.game_size)
        return self.get_player_observation()

    def get_player_observation(self):
        return np.expand_dims(np.concatenate((self.player_state, self.opponent_state)), 0).astype(np.float32)

    def get_opponent_observation(self):
        return np.expand_dims(np.concatenate((self.opponent_state, self.player_state)), 0).astype(np.float32)

    def agent_won(self):
        won_stacks = (self.player_state > self.opponent_state).astype('int')
        return 1 if np.sum(won_stacks) > self.game_size/2 else 0

    def get_observation_shape(self):
        return self.get_player_observation().shape

#This class is used to proff that the implemented RL Algorithm works for Continuous cases
class LunarLanderC_POC():
    def __init__(self, opponent_actor):
        #doesnt need opponent_actor
        self.env = gym.make('LunarLanderContinuous-v2')
        self.env._max_episode_steps = 800
        self.reward_sum = 0

    def step(self, action):
        action = np.squeeze(action)
        o, r, d, _ = self.env.step(action)
        r = r
        o = np.expand_dims(o, 0)
        o.astype(np.float32)
        self.reward_sum = self.reward_sum + r
        return o,r,d,None

    def agent_won(self):
        return self.reward_sum

    def reset(self):
        o = self.env.reset()
        o = np.expand_dims(o, 0)
        o.astype(np.float32)
        return o


    def get_observation_shape(self):
        return (1,) + self.env.observation_space.shape

#This class is used to proff that the implemented RL Algorithm works for Continuous cases
class LunarLander_POC():
    def __init__(self, opponent_actor):
        #doesnt need opponent_actor
        self.env = gym.make('LunarLander-v2')
        self.reward_sum = 0

    def step(self, action):
        action = np.squeeze(action)
        o, r, d, _ = self.env.step(action)
        o = np.expand_dims(o, 0)
        o.astype(np.float32)
        self.reward_sum = self.reward_sum + r
        return o,r,d,None

    def agent_won(self):
        return self.reward_sum

    def reset(self):
        o = self.env.reset()
        o = np.expand_dims(o, 0)
        o.astype(np.float32)
        return o


    def get_observation_shape(self):
        return (1,) + self.env.observation_space.shape

class WalkerC_POC():
    def __init__(self, opponent_actor):
        #doesnt need opponent_actor
        self.env = gym.make('BipedalWalker-v3')
        self.reward_sum = 0

    def step(self, action):
        action = np.squeeze(action)
        o, r, d, _ = self.env.step(action)
        o = np.expand_dims(o, 0)
        o.astype(np.float32)
        self.reward_sum = self.reward_sum + r
        return o,r,d,None

    def agent_won(self):
        return self.reward_sum

    def reset(self):
        o = self.env.reset()
        o = np.expand_dims(o, 0)
        o.astype(np.float32)
        return o


    def get_observation_shape(self):
        return (1,) + self.env.observation_space.shape
