import numpy as np

class Vier_gewinnt_gym:
    def __init__(self, opponent_actor):
        self.height = 6
        self.width = 7
        self.size = (self.width, self.height)
        self.max_steps = self.height*self.width
        self.step_counter = 0
        self.field = np.zeros(self.size)
        self.filled_levels = [0 for _ in range(self.width)]
        self.player_id = 1
        self.opponent_id = -1
        self.opponent_actor = opponent_actor
        self.opponent_invalid_move = False
        self.actor_invalid_move = False
        self.result_is_tied = False
        self.opponent_sign = -1
        self.actor_sign = 1
        self.win_reward = 1.
        self.loss_reward = -1.
        self.tie_reward = 0
        self.step_reward = 0.
        #TODO implement ties!
        self.tie_reward = 0.
        #toin coss
        if self.opponent_actor!=None:
            if np.random.rand() < .5:
                self.opponent_step()

    def reset(self):
        self.step_counter = 0
        self.field = np.zeros(self.size)
        self.filled_levels = [0 for _ in range(self.width)]
        self.opponent_invalid_move = False
        self.actor_invalid_move = False
        self.result_is_tied = False
        if self.opponent_actor!=None:
            if np.random.rand() < .5:
                self.opponent_step()

        return self.get_player_observation()

    def step(self, action):
        self.step_counter = self.step_counter + 1
        #print(np.flip(np.transpose(self.field), 0), action, self.step_counter)
        action = action[0]
        #actor allways goes first in step, first step by opponent is handled in init
        assert action <= self.width
        assert action >= 0
        #player makes invalid move
        if self.filled_levels[action] >= self.height:
            print(np.flip(np.transpose(self.field), 0), action, self.step_counter, 'player_invalid_move')
            self.actor_invalid_move = True
            self.agent_won_reward = self.loss_reward
            return self.get_player_observation(), self.loss_reward, True, None
        #player makes valid move
        else:
            self.field[action, self.filled_levels[action]] = self.actor_sign
            self.filled_levels[action] = self.filled_levels[action] + 1
        #player makes a winning move
        if self.check_matched_four(action, self.filled_levels[action]-1, self.actor_sign):
            print(np.flip(np.transpose(self.field), 0), action, self.step_counter, 'player winning move')
            self.agent_won_reward = self.win_reward
            return self.get_player_observation(), self.win_reward, True, None
        #player move makes game a tie
        elif self.is_tie():
            print(np.flip(np.transpose(self.field), 0), action, self.step_counter, 'player tied game')
            self.result_is_tied = True
            self.agent_won_reward = self.tie_reward
            return self.get_player_observation(), self.tie_reward, True, None

        #opponent moves second
        opponent_won = self.opponent_step()

        #opponent made invalid move
        if self.opponent_invalid_move:
            print(np.flip(np.transpose(self.field), 0), action, self.step_counter, 'opponent_invalid_move')
            self.agent_won_reward = self.win_reward
            return self.get_player_observation(), self.win_reward, True, None

        #opponent made winning_move
        if opponent_won:
            print(np.flip(np.transpose(self.field), 0), action, self.step_counter, 'opponent_winning_move')
            self.agent_won_reward = self.loss_reward
            return self.get_player_observation(), self.loss_reward, True, None

        #opponent move makes game a tie
        if self.is_tie():
            print(np.flip(np.transpose(self.field), 0), action, self.step_counter, 'opponent_tie')
            self.agent_won_reward = self.tie_reward
            return self.get_player_observation(), self.win_reward, True, None

        #else game has not ended, game goes on
        return self.get_player_observation(), self.step_reward, False, None

    def opponent_step(self):
        opponent_action, _ = self.opponent_actor.act(self.get_opponent_observation())
        opponent_action = opponent_action[0]
        if self.filled_levels[opponent_action]>=self.height:
            self.opponent_invalid_move = True
        else:
            self.field[opponent_action, self.filled_levels[opponent_action]] = self.opponent_sign
            self.filled_levels[opponent_action] = self.filled_levels[opponent_action] + 1
        opponent_won = self.check_matched_four(opponent_action, self.filled_levels[opponent_action]-1, self.opponent_sign)
        return opponent_won

    def check_matched_four(self, x, y, sign):
        neighbour_directions = [(1,0), (1,1), (0,1), (-1,1)]
        for direction in neighbour_directions:
            if self.check_neighbours_in_dir(x,y,sign,direction)+1 >=4:
                return True
        return False

    def check_neighbours_in_dir(self, x, y, sign, direction):
        dx, dy = direction
        len_found = 0

        #search dir direction
        search_on = True
        x_search = x
        y_search = y
        while search_on:
            x_search = x_search + dx
            y_search = y_search + dy
            search_on = self.check_neighbour(x_search, y_search, sign)
            if search_on:
                len_found = len_found + 1

        #search dir opposing direction
        dx = dx*-1
        dy = dy*-1
        x_search = x
        y_search = y
        search_on = True
        while search_on:
            x_search = x_search + dx
            y_search = y_search + dy
            search_on = self.check_neighbour(x_search, y_search, sign)
            if search_on:
                len_found = len_found + 1

        return len_found

    def check_neighbour(self, x, y, sign):
        if self.is_in_bounds(x,y):
            if self.field[x,y] == sign:
                return True
        else:
            return False

    def is_in_bounds(self, x, y):
        if x >= 0 and x <= self.width - 1 :
            if y >= 0 and y <= self.height - 1 :
                return True
        else:
            return False

    def get_player_observation(self):
        return np.reshape(self.field, (1,self.width*self.height))

    def get_opponent_observation(self):
        return np.reshape(self.field * -1., (1,self.width*self.height))

    def get_observation_shape(self):
        return (1,self.width*self.height)

    def is_tie(self):
        assert sum(self.filled_levels) <= self.height*self.width
        return sum(self.filled_levels) == self.height * self.width

    def agent_won(self):
        return self.agent_won_reward
