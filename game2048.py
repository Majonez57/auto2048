import random
import sys
import gym
import numpy as np
from gym import spaces
from PyQt5.QtWidgets import QApplication
import matplotlib.pyplot as plt
from math import log2
from gameWindow import GameWindow
from time import sleep

GAMEOVER   = 0
NOMOVE     = 1 #make sure to not reward nothing moves
SCOREDMOVE = 2 #move added score

def moving_average(data, window_size):
    # Pad the data at the beginning to handle edges
    padded_data = np.concatenate([np.zeros(window_size-1), data])
    
    # Create an array to hold the moving average values
    moving_avg = np.zeros(len(data))
    
    # Compute the moving average
    for i in range(len(data)):
        moving_avg[i] = np.mean(padded_data[i:i+window_size])
    
    return moving_avg


class Game2048Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps" : 4}

    def __init__(self, render_mode=None):

        # Observations are from 1->15, as there 
        self.observation_space = spaces.Box(low=0, high=16, shape=(4,4), dtype=int)

        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == 'human':
            self.app = QApplication(sys.argv)
            self.game_window = GameWindow()

        self.score = 0
        self.terms = 0
        self.avg = 0
        self.allresults = []
        self.fitness = 0
        self.allfitness = []

    def _get_obs(self):
            return [[0 if x == 0 else log2(x) for x in y] for y in self.board]
    
    def _get_info(self):
            return {"score": self.score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        
        self.allresults.append(self.score)
        self.allfitness.append(self.fitness)
        
        self.board = [[0 for _ in range(4)] for _ in range(4)]
        self.score = 0
        self.fitness = 0

        self.add_random_tile()
        self.add_random_tile()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        moves = ['up', 'down', 'left', 'right']

        init_empty_cells = sum([1 for i in range(4) for j in range(4) if self.board[i][j] == 0]) # Empty Cells pre-move

        result = self.move(moves[action])

        empty_cells = sum([1 for i in range(4) for j in range(4) if self.board[i][j] == 0]) # Empty Cells pre-move

        delta_space = init_empty_cells - sum([1 for i in range(4) for j in range(4) if self.board[i][j] == 0]) # Empty Cells post-move
        delta_space += 1 # To account for the new block added

        terminated = result == GAMEOVER

        if terminated:
            self.terms += 1
            self.avg += self.score
            if self.terms == 20:
                print("Average Score in last 20 runs", self.avg/self.terms)
                self.terms = 0
                self.avg = 0

        # Reward
        # We want to reward the agent for having as few boxes as possible
        # And punish it when it adds more boxes
        # However we also want to make sure non-moves and terminal moves are also not rewarded
        if not self.is_move_possible():
            reward = 0
            # Making a move that leads to a terminal state
        elif result == terminated:
            reward = 0 
            # Do not reward the invalid terminal move
        elif result == NOMOVE:
            reward = 0
            # Penalize moves that do nothing
        else:
            R = 0.9
            
            delta_space /= 16 # Normalize to [0, 1]

            # result cannot be 0 so will not be math error
            result = log2(result) / 16 # Normalize to [0,1]
            
            reward = R * delta_space + (1-R) * result
            self.fitness += reward
            # This will balance reducing the space on the board
            # And trying to score
            # And penalize adding to the board

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.game_window.update_board(self.board, self.score, self.fitness)
            self.app.processEvents()
            sleep(1 / 4) #4 fps
    
        return observation, reward, terminated, False, info

    def render(self):
        pass
    
    def plot_learning_curve(self):

        fig, ax = plt.subplots()

        ax.scatter(range(len(self.allresults)), self.allresults, marker='x', c='red')
        ax.tick_params(axis='y', labelcolor='red') 
        ax.set_ylabel('Episodic Score')
        
        window_size = 20 # 20 Episode moving average

        ax.plot(moving_average(self.allresults, window_size), c='green', linestyle='--')

        ax.set_xlabel('Episode')
        plt.show()

    def close(self):
        if self.window is not None:
            self.window.close()

    def add_random_tile(self):
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.board[i][j] == 0]
        if empty_cells:
            i, j = self.np_random.choice(empty_cells)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def move(self, direction):
        moved = False
        score_add = 0

        if direction == 'up':
            for j in range(4):
                new_column = []
                for i in range(4):
                    if self.board[i][j] != 0:
                        new_column.append(self.board[i][j])
                new_column, score = self.merge_tiles(new_column)
                for i in range(4):
                    if i < len(new_column):
                        if self.board[i][j] != new_column[i]:
                            moved = True
                        self.board[i][j] = new_column[i]
                    else:
                        self.board[i][j] = 0
                score_add += score

        elif direction == 'down':
            for j in range(4):
                new_column = []
                for i in range(3, -1, -1):
                    if self.board[i][j] != 0:
                        new_column.append(self.board[i][j])
                new_column.reverse()
                new_column, score = self.merge_tiles(new_column)
                for i in range(3, -1, -1):
                    if i >= 4 - len(new_column):
                        if self.board[i][j] != new_column[i - (4 - len(new_column))]:
                            moved = True
                        self.board[i][j] = new_column[i - (4 - len(new_column))]
                    else:
                        self.board[i][j] = 0
                score_add += score

        elif direction == 'left':
            for i in range(4):
                new_row = []
                for j in range(4):
                    if self.board[i][j] != 0:
                        new_row.append(self.board[i][j])
                new_row, score = self.merge_tiles(new_row)
                for j in range(4):
                    if j < len(new_row):
                        if self.board[i][j] != new_row[j]:
                            moved = True
                        self.board[i][j] = new_row[j]
                    else:
                        self.board[i][j] = 0
                score_add += score

        else:  # right
            for i in range(4):
                new_row = []
                for j in range(3, -1, -1):
                    if self.board[i][j] != 0:
                        new_row.append(self.board[i][j])
                new_row.reverse()
                new_row, score = self.merge_tiles(new_row)
                for j in range(3, -1, -1):
                    if j >= 4 - len(new_row):
                        if self.board[i][j] != new_row[j - (4 - len(new_row))]:
                            moved = True
                        self.board[i][j] = new_row[j - (4 - len(new_row))]
                    else:
                        self.board[i][j] = 0
                score_add += score


        self.score += score_add
        if moved:
            self.add_random_tile()
            return score_add if score_add > 0 else NOMOVE
        else:
            if not self.is_move_possible():
                return GAMEOVER
            return NOMOVE

    def merge_tiles(self, row):
        new_row = []
        skip = False
        score = 0
        for i in range(len(row)):
            if not skip:
                if i < len(row) - 1 and row[i] == row[i + 1]:
                    new_row.append(row[i] * 2)
                    skip = True
                    score += row[i] * 2
                else:
                    new_row.append(row[i])
            else:
                skip = False
        return new_row, score

    def is_move_possible(self):
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    return True

        for i in range(3):
            for j in range(3):
                if self.board[i][j] == self.board[i][j + 1] or self.board[i][j] == self.board[i + 1][j]:
                    return True

        for i in range(3):
            if self.board[i][3] == self.board[i + 1][3]:
                return True

        for j in range(3):
            if self.board[3][j] == self.board[3][j + 1]:
                return True

        return False