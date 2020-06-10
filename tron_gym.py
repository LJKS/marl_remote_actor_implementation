import numpy as np
import random

class Tron_gym:
    def __init__(self, opponent=None, size=(12,12)):
        self.fail_reward = -10
        self.win_reward = 10
        self.step_reward = -0.01
        self.action_dict = {0:(1,0),1:(0,1),2:(0,-1),3:(-1,0)}
        self.size = size
        self.x_size, self.y_size = size
        self.player_pos = self.generate_random_pos()
        self.opponent_pos = ((self.x_size -1)- self.player_pos[0], (self.y_size-1) - self.player_pos[1])
        self.opponent = opponent
        self.agent_won_flag_flag = None
        self.wall_map = np.zeros(self.size)
        self.active_player_marker = 1
        self.passive_player_marker = -1
        # fifty-fifty chance of player vs. opponent beginning the game
        # do not run this if we only build the env with None opponent for estimating action and observation shapes
        if opponent != None:
            if np.random.randint(0,2) == 1:
                opponent_action, _ = self.opponent.act(self.get_opponent_observation())
                opponent_move_valid = self.opponent_acts(opponent_action)
                #no losing in first step, would generate weird code behaviour subsequently
                if not opponent_move_valid:
                    self.__init__(opponent, size)


    def reset(self):
        self.__init__(self.opponent, self.size)
        return self.get_player_observation()

    def step(self, action):
        #implements a step by player and opponent; in init sometimes opponent made a move first . in step allways player goes first
        player_move_valid = self.player_acts(action)
        if not player_move_valid:
            self.agent_won_flag = self.fail_reward
            return self.get_player_observation(), self.fail_reward, True, None
        else:
            opponent_action,_ = self.opponent.act(self.get_opponent_observation())
            opponent_move_valid = self.opponent_acts(opponent_action)
            if not opponent_move_valid:
                self.agent_won_flag = self.win_reward
                return self.get_player_observation(), self.win_reward, True, None
            else:
                return self.get_player_observation(), self.step_reward, False, None

    def generate_random_pos(self):
        return (random.randint(0, self.x_size-1), random.randint(0,self.y_size-1))

    def is_valid_move(self, target_pos):
        if self.is_valid_pos(target_pos):
            return (self.wall_map[target_pos]==0) and (not (target_pos in (self.player_pos, self.opponent_pos)))
        else:
            return False

    def is_valid_pos(self, target_pos):
        if (target_pos[0] < 0) or (target_pos[1] < 0) :
            return False
        elif (target_pos[0] >= self.size[0]) or (target_pos[1] >= self.size[1]) :
            return False
        else:
            return True

    def move_target_pos(self, start, action):
        move = self.action_dict[int(action)]
        target_pos = (start[0] + move[0], start[1] + move[1])
        return target_pos

    def opponent_acts(self, action):
        #returns True iff opponent does not lose, else returns False
        target_pos = self.move_target_pos(self.opponent_pos, action)
        valid_move = self.is_valid_pos(target_pos)
        #move was invalid
        if not valid_move:
            return False
        else:
            self.wall_map[self.opponent_pos] = 1
            self.opponent_pos = target_pos
            return True


    def player_acts(self, action):
        #returns True iff player does not lose, else returns False
        target_pos = self.move_target_pos(self.player_pos, action)
        valid_move = self.is_valid_pos(target_pos)
        #move was invalid
        if not valid_move:
            return False
        else:
            self.wall_map[self.player_pos] = 1
            self.player_pos = target_pos
            return True

    def get_player_observation(self):
        agent_map = np.zeros(self.size)
        agent_map[self.player_pos] = self.active_player_marker
        agent_map[self.opponent_pos] = self.passive_player_marker
        return np.expand_dims(np.stack((agent_map, self.wall_map), axis=-1), 0)

    def get_opponent_observation(self):
        agent_map = np.zeros(self.size)
        agent_map[self.opponent_pos] = self.active_player_marker
        agent_map[self.player_pos] = self.passive_player_marker
        return np.expand_dims(np.stack((agent_map, self.wall_map), axis=-1), 0)

    def get_observation_shape(self):
        return self.get_player_observation().shape

    def get_action_size(self):
        return 4

    def agent_won(self):
        return self.agent_won_flag
