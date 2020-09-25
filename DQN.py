from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
import random
import numpy as np
import pandas as pd
from operator import add
import collections



class DQNAgent(object):
    def __init__(self, params):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.model = self.network()

    def network(self):
        model = Sequential()
        model.add(Conv2D(28, (4, 4), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((4, 4)))
        model.add(Flatten())
        model.add(Dense(self.first_layer, activation='tanh'))
        model.add(Dense(self.second_layer, activation='tanh'))
        model.add(Dense(self.third_layer, activation='tanh'))
        # model.add(Dense(output_dim=self.third_layer, activation='tanh'))
        # model.add(Dense(output_dim=self.third_layer, activation='tanh'))
        model.add(Dense(3, activation='linear'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if self.load_weights:
            model.load_weights(self.weights)
        return model
    
    def get_state(self, game, player, food):
        
        initial = np.zeros(shape=(28,28,1))
        for i in range(22):
            if(i!=0 and i!=21):
                initial[i+3][3] = [-10]
                initial[i+3][24] = [-10]
            else:
                for j in range(22):
                    if(i==0 or i==21):
                        initial[i+3][j+3] = [-10]
        # print(food.x_food)
        initial[(food.x_food//20)+3][(food.y_food//20)+3] = [10]
        for p in player.position:
            initial[(int(p[0])//20)+3][(int(p[1])//20)+3] = [-10]
        initial[(int(player.position[-1][0])//20)+3][(int(player.position[-1][1])//20)+3] =[ 1]
        # for i in range(22):
        #     print(initial[i])
        state =initial
        # state = [
        #     (player.x_change == 20 and player.y_change == 0 and ((list(map(add, player.position[-1], [20, 0])) in player.position) or
        #     player.position[-1][0] + 20 >= (game.game_width - 20))) or (player.x_change == -20 and player.y_change == 0 and ((list(map(add, player.position[-1], [-20, 0])) in player.position) or
        #     player.position[-1][0] - 20 < 20)) or (player.x_change == 0 and player.y_change == -20 and ((list(map(add, player.position[-1], [0, -20])) in player.position) or
        #     player.position[-1][-1] - 20 < 20)) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add, player.position[-1], [0, 20])) in player.position) or
        #     player.position[-1][-1] + 20 >= (game.game_height-20))),  # danger straight

        #     (player.x_change == 0 and player.y_change == -20 and ((list(map(add,player.position[-1],[20, 0])) in player.position) or
        #     player.position[ -1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],
        #     [-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == -20 and player.y_change == 0 and ((list(map(
        #     add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
        #     (list(map(add,player.position[-1],[0,20])) in player.position) or player.position[-1][
        #      -1] + 20 >= (game.game_height-20))),  # danger right

        #      (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],[20,0])) in player.position) or
        #      player.position[-1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == -20 and ((list(map(
        #      add, player.position[-1],[-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
        #     (list(map(add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (
        #     player.x_change == -20 and player.y_change == 0 and ((list(map(add,player.position[-1],[0,20])) in player.position) or
        #     player.position[-1][-1] + 20 >= (game.game_height-20))), #danger left

        #     player.x_change == -20,  # move left
        #     player.x_change == 20,  # move right
        #     player.y_change == -20,  # move up
        #     player.y_change == 20,  # move down
        #     food.x_food < player.x,  # food left
        #     food.x_food > player.x,  # food right
        #     food.y_food < player.y,  # food up
        #     food.y_food > player.y  # food down
        #     ]

        # for i in range(len(state)):
        #     if state[i]:
        #         state[i]=1
        #     else:
        #         state[i]=0

        return state

    def set_reward(self, player, crash):
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
        target_f = self.model.predict(np.array([state]))
        target_f[0][np.argmax(action)] = target
        self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)