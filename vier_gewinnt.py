import numpy as np

class Vier_gewinnt_gym:
    def __init__(self. opponent_actor):
        self.height = 6
        self.width = 7
        self.size = (self.width, self.height)
        self.max_steps = self.height*self.width
        self.step = 0
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
        self.step_reward = 0.
        #TODO implement ties!
        self.tie_reward = 0.
        #toin coss
        if np.random.rand < .5:
            self.opponent_step()

        def step(self, action):
            #actor allways goes first in step, first step by opponent is handled in init
            assert action <= self.width
            assert action >= 0
            #player makes invalid move
            if self.filled_levels[action] >= self.height:
                self.actor_invalid_move = True
                return self.get_player_observation(), self.loss_reward, True, None
            #player makes valid move
            else:
                self.field[action, self.filled_levels[action]] = self.actor_sign
                self.filled_levels[action] = self.filled_levels[action] + 1
            #player makes a winning move
            if self.check_matched_four(action, self.filled_levels[action]-1, self.actor_sign)
                return self.get_player_observation, self.win_reward, True, None
            #player move makes game a tie
            elif self.is_tie():
                self.result_is_tied = True
                return self.get_player_observation, self.tie_reward, True, None

            #opponent moves second
            opponent_won = self.opponent_step()

            #opponent made invalid move
            if self.opponent_invalid_move:
                return self.get_player_observation, self.win_reward, True, None

            #opponent made winning_move
            if opponent_won:
                return self.get_player_observation, self.loss_reward, True, None

            #opponent move makes game a tie
            if self.is_tie():
                return self.get_player_observation, self.win_reward, True, None

            #else game has not ended, game goes on
            return self.get_player_observation, self.step_reward, False, None

        def opponent_step(self):
            opponent_action, _ = self.opponent.act(self.get_opponent_observation())
            if self.filled_levels[opponent_action]>=self.height:
                self.opponent_invalid_move = True
            else:
                self.field[opponent_action, self.filled_levels[opponent_action]] = self.opponent_sign
                self.filled_levels[opponent_action] = self.filled_levels[opponent_action] + 1
            opponent_won = self.check_matched_four(opponent_action, self.filled_levels[opponent_action]-1, self.opponent_sign)
            return opponent_won

        def check_matched_four(self, x, y, sign):
            neighbour_directions = [(1,0), (1,1), (0,1), (-1,1)]
            found_match = False
            for dir in neighbour_directions:
                if self.check_neighbours_in_dir(x,y,sign,dir)+1 >=4:
                    return True
            return False

        def check_neighbours_in_dir(self, x, y, sign, direction):
            dx, dy = dir
            len_found = 0

            #search dir direction
            search_on = True
            while search_on:
                x = x + dx
                y = y + dy
                search_on = self.check_neighbour(x, y, sign)
                if search_on:
                    len_found = len_found + 1

            #search dir opposing direction
            dx = dx*-1
            dy = dy*-1
            search_on = True
            while search_on:
                x = x + dx
                y = y + dy
                search_on = self.check_neighbour(x, y, sign)
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
            return sum(self.filled_levels) == self.height * self.width
