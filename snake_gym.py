import random
import numpy as np
import math

class Snake_gym:
    def __init__(self, opponent=None, size=(12,12)):
        self.fail_reward = -100
        self.food_reward = 10
        self.step_reward = -0.01
        self.win_reward = self.food_reward
        self.action_dict = {0:(1,0),1:(0,1),2:(0,-1),3:(-1,0)}
        self.size = size
        self.x_size, self.y_size = size
        self.snake = [self.generate_random_pos()]
        self.food = []
        self.generate_food()
        self.final_len = self.reduce_mult(self.size) / 2

    def reset(self):
        self.snake = [self.generate_random_pos()]
        self.food = []
        self.generate_food()
        return self.get_observation()

    def generate_random_pos(self):
        return (random.randint(0, self.x_size-1), random.randint(0,self.y_size-1))

    def generate_food(self):
        food_not_generated = True
        while food_not_generated:
            pos = self.generate_random_pos()
            if (pos not in self.snake) and (pos not in self.food):
                self.food.append(pos)
                food_not_generated = False

    def snake_head(self):
        return self.snake[0]

    def snake_length(self):
        return len(self.snake)

    def get_observation(self):
        snake_map = np.zeros(self.size)
        food_map = np.zeros(self.size)
        for i, pos in enumerate(self.snake):
            j = i+1
            snake_map[pos]=j
        for pos in self.food:
            food_map[pos]=1
        return np.expand_dims(np.stack((snake_map, food_map), axis=-1), 0)

    def is_valid_move(self, target_pos):
        if target_pos in self.snake:
            return False
        elif (target_pos[0] < 0) or (target_pos[1] < 0) :
            return False
        elif (target_pos[0] >= self.size[0]) or (target_pos[1] >= self.size[1]) :
            return False
        else:
            return True


    def step(self, action):
        action = int(action)
        #debug
        x = self.get_observation()
        #print(np.sum(np.squeeze(x), axis=-1))
        #print('action:', action)
        #end debug
        action = self.action_dict[action]
        go_to_pos = (self.snake_head()[0]+action[0], self.snake_head()[1]+action[1])
        if self.is_valid_move(go_to_pos):
            self.snake.insert(0,go_to_pos)
            #if find food
            if go_to_pos in self.food:
                #remove food, its eaten now
                self.food.remove(go_to_pos)
                #if got to max snake length
                if self.snake_length() +1 >= self.final_len :
                    return self.get_observation(), self.win_reward, True, None
                #else just plant new food and continue
                else:
                    self.generate_food()
                    return self.get_observation(), self.food_reward, False, None
            #if not find food remove snake tail, as it moved one forward and didnt grow
            else:
                self.snake.pop()
                return self.get_observation(), self.step_reward, False, None
        #else is fail
        else:
            return self.get_observation(), self.fail_reward, True, None

    def get_observation_shape(self):
        #batch_size 1, intermediate board size, 2 channels for snake and food
        return (1,) + self.size + (2,)

    def get_action_size(self):
        return 4

    def reduce_mult(self, prod_list):
        prod = 1
        for elem in prod_list:
            prod = prod*elem
        return prod

    def agent_won(self):
        return int(len(self.snake) >= self.final_len)
