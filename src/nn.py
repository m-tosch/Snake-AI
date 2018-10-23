from snake_game import SnakeGame
from random import randint
import numpy as np
import tflearn
import math
import tflearn.layers.core
import tflearn.layers.estimator
from statistics import mean
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#disables warning "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"


class SnakeNN:
    def __init__(self, initial_games = 100, test_games = 100, max_steps = 1500, lr = 0.0005, filename = 'snake_nn.tflearn'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.max_steps = max_steps
        self.lr = lr
        self.filename = filename
        self.vectors_and_keys = [
                [[-1, 0], 0], # LEFT
                [[0, 1], 1],  # UP
                [[1, 0], 2],  # RIGHT
                [[0, -1], 3]] # DOWN
        self.neurons = 250
        self.batch_size = 8
        self.epochs = 3

    def initial_population(self, num_games):
        print('--- initial_population ---')
        start = time.time()
        training_data = []
        for i in range(num_games):
            game = SnakeGame()
            _, prev_score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food) # [1/0, 1/0, 1/0, angle] obstacle left, front, right + food angle
            prev_food_distance = self.get_food_distance(snake, food)
            for j in range(self.max_steps):
                action, game_action = self.generate_action(snake) # action -1/0/1  game_action 0/1/2/3
                done, score, snake, food = game.step(game_action)
                if done:
                    # left,forward,right | obst.left,front,right | angle | survived
                    # [ array([-1/0/1, 1/0, 1/0, 1/0 -1-to-1]), -1/0/1 ]
                    training_data.append([self.add_action_to_observation(action, prev_observation), -15]) # -1 snake didn't survive
                    break
                else:
                    food_distance = self.get_food_distance(snake, food) # always >=1
                    if score > prev_score or food_distance < prev_food_distance:
                        training_data.append([self.add_action_to_observation(action, prev_observation), 1]) # 1 snake survived and right direction
                    else:
                        training_data.append([self.add_action_to_observation(action, prev_observation), 0]) # 0 snake survived but wrong direction
                    prev_observation = self.generate_observation(snake, food)
                    prev_food_distance = food_distance
                    prev_score = score
            print(' game: ' + str(i+1) + '/' + str(num_games) + ' [' + str(round(((i+1)/num_games)*100,1)) + '%]' +' max_steps: ' + str(self.max_steps) + ' time: ' + str(round(time.time() - start, 3)) + 's', end='\r')
        end = time.time()
        print(' game: ' + str(i+1) + '/' + str(num_games) + ' [' + str(round(((i+1)/num_games)*100,1)) + '%]' +' max_steps: ' + str(self.max_steps) + ' time: ' + str(round(time.time() - start, 3)) + 's')
        return training_data

    def generate_action(self, snake):
        action = randint(0,2) - 1 # -1 (left), 0 (forward), 1 (right)
        return action, self.get_game_action(snake, action)

    def get_game_action(self, snake, action):
        snake_direction = self.get_snake_direction_vector(snake)
        new_direction = snake_direction
        if action == -1:
            new_direction = self.turn_vector_to_the_left(snake_direction)
        elif action == 1:
            new_direction = self.turn_vector_to_the_right(snake_direction)
        # new_direction [-1 left, 0 forward, 1 right]
        for pair in self.vectors_and_keys:
            if pair[0] == new_direction.tolist():
                game_action = pair[1]
        return game_action # [0 up, 1 right, 2 down, 3 left]

    def generate_observation(self, snake, food):
        snake_direction = self.get_snake_direction_vector(snake)
        food_direction = self.get_food_direction_vector(snake, food)
        barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        barrier_front = self.is_direction_blocked(snake, snake_direction)
        barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        angle = self.get_angle(snake_direction, food_direction)
        return np.array([int(barrier_left), int(barrier_front), int(barrier_right), angle])

    def add_action_to_observation(self, action, observation):
        return np.append([action], observation)

    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_food_direction_vector(self, snake, food):
        return np.array(food) - np.array(snake[0])

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def get_food_distance(self, snake, food):
        return np.linalg.norm(self.get_food_direction_vector(snake, food))

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def model(self):
        network = tflearn.layers.core.input_data(shape=[None, 5, 1], name='input')
        network = tflearn.layers.core.fully_connected(network, self.neurons, activation='relu')
        network = tflearn.layers.core.fully_connected(network, 1, activation='linear')
        network = tflearn.layers.estimator.regression(network, optimizer='adam', learning_rate=self.lr, batch_size=self.batch_size, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        print('--- train_model ---')
        start = time.time()
        x = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(x,y, n_epoch = self.epochs, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        end = time.time()
        print(time.strftime("Time elapsed: %H:%M:%S", time.gmtime(end - start)))
        return model

    def test_model(self, model, n):
        print('--- test_model ---')
        start = time.time()
        steps_arr = [0]
        scores_arr = [0]
        for i in range(round(self.test_games / n)):
            steps = 0
            game = SnakeGame()
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.max_steps):
                predictions = []
                for action in range(-1, 2):
                   predictions.append(model.predict(self.add_action_to_observation(action, prev_observation).reshape(-1, 5, 1)))
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                done, score, snake, food  = game.step(game_action)
                if done:
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
            print('game: ' + str(i+1) + '/' + str(round(self.test_games / n)) + ' steps_avg=' + str(round(mean(steps_arr), 2)) + ' score_avg=' + str(round(mean(scores_arr), 2)), end='\r')
        end = time.time()
        avg_steps = mean(steps_arr)
        avg_score = mean(scores_arr)
        print('game: ' + str(i+1) + '/' + str(round(self.test_games / n)) + ' steps_avg=' + str(round(mean(steps_arr), 2)) + ' score_avg=' + str(round(mean(scores_arr), 2)))
        print('steps: avg=' + str(round(avg_steps, 2)) + ' max=' + str(max(steps_arr)) + ' min=' + str(min(steps_arr)))
        print('score: avg=' + str(round(avg_score, 2)) + ' max=' + str(max(scores_arr)) + ' min=' + str(min(scores_arr)))
        print(time.strftime("Time elapsed: %H:%M:%S", time.gmtime(end - start)))
        return avg_steps, avg_score

    def visualise_game(self, model):
        game = SnakeGame(gui = True)
        _, score, snake, food = game.start()
        prev_observation = self.generate_observation(snake, food)
        for _ in range(self.max_steps):
            predictions = []
            for action in range(-1, 2):
               predictions.append(model.predict(self.add_action_to_observation(action, prev_observation).reshape(-1, 5, 1)))
            action = np.argmax(np.array(predictions))
            game_action = self.get_game_action(snake, action - 1)
            done, score, snake, food  = game.step(game_action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(snake, food)
        game.end_game()
        print('snake: ' + str(snake))
        print('food: ' + str(food))
        print('prev_obs: ' + str(prev_observation))
        print('score: ' + str(score))

    def train(self):
        training_data = self.initial_population(self.initial_games)
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test(nn_model)

    def visualise(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualise_game(nn_model)

    def test_(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test(nn_model)

    def test(self, nn_model):
        print('--- test ---')
        start = time.time()
        avg_steps_arr = []
        avg_score_arr = []
        n = 5
        for i in range(n):
            print(str(i+1) + '/' + str(n))
            avg_steps, avg_score = self.test_model(nn_model, n)
            avg_steps_arr.append(avg_steps)
            avg_score_arr.append(avg_score)
        avg_steps = round(mean(avg_steps_arr),3)
        avg_score = round(mean(avg_score_arr),3)
        print('\ninitial_games: ' + str(self.initial_games) + " max_steps: " + str(self.max_steps))
        print('Summary from ' + str(self.test_games) + ' test games')
        print('avg. steps: ' + str(avg_steps) + ' avg. score: ' + str(avg_score))
        print(time.strftime("Time elapsed: %H:%M:%S", time.gmtime(time.time() - start)))
        #print('initial games: ' + str(self.initial_games) + ' Neurons: ' + str(self.neurons) + ' lr: ' + str(self.lr) + ' batch_size: ' + str(self.batch_size) + ' epochs: ' + str(self.epochs))
        return avg_steps, avg_score


if __name__ == "__main__":
    start = time.time()
    SnakeNN().train()
    #SnakeNN().test_()
    #SnakeNN().visualise()
    print('-----------------------')
    print(time.strftime("Total runtime: %H:%M:%S", time.gmtime(time.time() - start)))