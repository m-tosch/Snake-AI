from snake_game import SnakeGame
from random import randint
import numpy as np
import tflearn
import math
from statistics import mean
import time



class SnakeSearch:
    def __init__(self, test_games = 1000, goal_steps = 2000):
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.vectors_and_keys = [
                [[-1, 0], 0], # LEFT
                [[0, 1], 1],  # UP
                [[1, 0], 2],  # RIGHT
                [[0, -1], 3]] # DOWN

    def get_food_direction_vector(self, point, food):
        return np.array(food) - np.array(point)
    
    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_food_distance(self, point, food):
        return np.linalg.norm(self.get_food_direction_vector(point, food))

    def generate_observation(self, snake, food):
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake[0], food)
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, food_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])
    
    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])
    
    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi
    
    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def is_valid_cell(self, cell, queue, visited, game):
        if cell in queue or cell in visited:
            return False
        elif cell in game.snake[:-1]:
            return False
        elif cell[0] <= 0 or cell[0] > game.board["width"] or cell[1] <= 0 or cell[1] > game.board["height"]:
            return False
        else:
            return True

    def get_game_action(self, game):
        node = game.snake[0]
        cells = []
        cell_up = [node[0], node[1] + 1]
        cell_down = [node[0], node[1] - 1]
        cell_left = [node[0] - 1, node[1]]
        cell_right = [node[0] + 1, node[1]]
        for cell in [cell_up, cell_down, cell_left, cell_right]:
            if self.is_valid_cell(cell, [], [], game):
                food_distance = self.get_food_distance(cell, game.food)
                cells.append([cell, food_distance])
                cells.sort(key=lambda x: x[1])
        if len(cells) == 0:
            return 0 # snake is trapped, no way out. game over with next step
        dir = np.array(cells[0][0]) - np.array(node)
        for pair in self.vectors_and_keys:
            if pair[0] == dir.tolist():
                return pair[1]

    def test(self):
        print('--- test ---')
        start = time.time()
        steps_arr = [0]
        scores_arr = [0]
        for i in range(self.test_games):
            steps = 0
            game = SnakeGame()
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                game_action = self.get_game_action(game)
                done, score, snake, food  = game.step(game_action)
                if done:
                    if prev_observation[0] != 1 or prev_observation[1] != 1 or prev_observation[2] != 1:
                        action_str = 'UP'
                        if game_action == 1:
                            action_str = 'RIGHT'
                        elif game_action == 2:
                            action_str = 'DOWN'
                        elif game_action == 3:
                            action_str = 'LEFT'
                        print(str(i) + '/' + str(self.test_games) + ' ' + str(prev_observation) + ' ' + action_str + ' [' + str(round(mean(steps_arr), 2)) + ', ' + str(round(mean(scores_arr), 2)) + ']')
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)
                    steps += 1
            print('game: ' + str(i+1) + '/' + str(self.test_games) + ' [' + str(round(((i+1)/self.test_games)*100,1)) + '%]' +' goal_steps: ' + str(self.goal_steps) + ' time: ' + str(round(time.time() - start, 3)) + 's', end='\r')
            steps_arr.append(steps)
            scores_arr.append(score)
        end = time.time()
        avg_steps = mean(steps_arr)
        avg_score = mean(scores_arr)
        print('game: ' + str(i+1) + '/' + str(self.test_games) + ' [' + str(round(((i+1)/self.test_games)*100,1)) + '%]' + ' goal_steps: ' + str(self.goal_steps) + ' time: ' + str(round(end - start, 3)) + 's')
        print('steps: avg=' + str(round(avg_steps, 2)) + ' max=' + str(max(steps_arr)) + ' min=' + str(min(steps_arr)))
        print('score: avg=' + str(round(avg_score, 2)) + ' max=' + str(max(scores_arr)) + ' min=' + str(min(scores_arr)))
        print(time.strftime("Total time elapsed: %H:%M:%S", time.gmtime(end - start)))

    def visualise(self):
        game = SnakeGame(gui = True)
        game.start()
        for _ in range(self.goal_steps):
            game_action = self.get_game_action(game)
            done, score, snake, food = game.step(game_action)
            if done:
                break
        game.end_game()
        print('-----')
        print('snake: ' + str(snake))
        print('food: ' + str(food))
        print('score: ' + str(score))

if __name__ == "__main__":
    SnakeSearch().test()
    #SnakeSearch().visualise()