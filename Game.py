from Model import Model
import numpy as np
import random
import tensorflow as tf
from tensorflow_probability.python.distributions import Categorical


class GridWorld(Model):

    def __init__(self, map_size):
        super().__init__()
        self.map_size = map_size
        self.game_step = 0
        self.num_actions = 5
        self.state = np.zeros((self.map_size, self.map_size, 2))
        self.agent_x = -1
        self.agent_y = -1
        self.total_reward = 0
        self.actions = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]
        self.game_over = False
        self.reset()
        self.create_model(self.map_size, 2, self.num_actions)

    def reset(self):
        self.game_over = False
        self.game_step = 0
        self.total_reward = 0
        # create 8x8x2 grid of 0's and 3rd dim is if square is free or has a block in it
        self.state = np.zeros((self.map_size, self.map_size, 2))
        # agent starting coords
        self.agent_x = random.randint(0, self.map_size - 1)
        self.agent_y = random.randint(0, self.map_size - 1)
        self.state[self.agent_x, self.agent_y, 1] = 1
        # self.state[self.map_goal[0], self.map_goal[1]] = 10
        # add obstacles
        for i in range(self.map_size):
            x = random.randint(0, self.map_size - 1)
            y = random.randint(0, self.map_size - 1)
            self.state[x, y, 0] = random.randint(1, 5)
        # best solution
        for x, y in self.actions:
            self.state[x + 5, y + 5, 0] = random.randint(1, 5)

    def agent_move(self):
        state = tf.convert_to_tensor([self.state])
        probabilities = self.model.predict(state).flatten()
        # probabilities = probabilities / np.sum(probabilities)
        action_probabilities = Categorical(probs=probabilities)
        action_index = action_probabilities.sample().numpy()
        reward = -1
        # get our agents chosen action
        dx, dy = self.actions[action_index]
        # check x, y coords are legal
        if 0 <= self.agent_x + dx < self.map_size and 0 <= self.agent_y + dy < self.map_size:
            # remove agents current board state, turn square from 1 to 0
            self.state[self.agent_x, self.agent_y, 1] = 0
            # update agent state
            self.agent_x, self.agent_y = self.agent_x + dx, self.agent_y + dy
            # update board state
            self.state[self.agent_x, self.agent_y, 1] = 1
            # go through agent actions
            for dx, dy in self.actions:
                # update agent pos
                xx, yy = self.agent_x + dx, self.agent_y + dy
                # our reward is the sum of legal moves the agent can make? What is this? What is the goal?
                if 0 <= xx < self.map_size and 0 <= yy < self.map_size:
                    reward += self.state[xx, yy, 0]

        self.total_reward += reward

        # if self.game_step == 16 and self.total_reward > 0:
        #     self.rewards_memory.append(100)
        # else:
        #     self.rewards_memory.append(reward)
        self.rewards_memory.append(reward)

        self.game_step += 1
        self.game_over = self.game_step == 16
        self.states_memory.append(self.state)
        self.action_memory.append(action_index)


map_size = 10
game = GridWorld(map_size)
batch_size = 1
number_of_batches = 2000

EPISODES = []
REWARDS = []
AGENT_STATES, AGENT_STEPS, AGENT_POSITIONS, AGENT_REWARDS = [], [], [], []
running_reward = 0

for batch in range(number_of_batches):

    for episode in range(batch_size):
        if batch + 1 == number_of_batches:
            while not game.game_over:
                game.agent_move()
                state = np.zeros((game.map_size, game.map_size), dtype=int)
                for y in range(game.map_size):
                    for x in range(game.map_size):
                        if game.state[x, y, 1] == 0:
                            state[game.map_size - 1 - y][x] = 0 if game.state[x, y, 0] == 0 else 10
                        else:
                            state[game.map_size - 1 - y][x] = 5 if game.state[x, y, 0] == 0 else 100
                AGENT_STATES.append(state)
                AGENT_STEPS.append(game.game_step)
                AGENT_POSITIONS.append((game.agent_x, game.agent_y))
                AGENT_REWARDS.append(int(game.rewards_memory[-1]))
        else:
            while not game.game_over:
                game.agent_move()

    running_reward = 0.05 * np.sum(game.rewards_memory) + (1 - 0.05) * running_reward

    if batch % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, batch))

    REWARDS.append(running_reward)
    EPISODES.append(batch)
    game.train_model()
    game.states_memory = []
    game.action_memory = []
    game.rewards_memory = []
    game.reset()

from data_gui import DataGUI
_gui = DataGUI(np.array(AGENT_STATES), AGENT_STEPS, AGENT_POSITIONS, AGENT_REWARDS, [EPISODES, REWARDS],
               map_size)
_gui.run_gui()

# import matplotlib.pyplot as plt
#
# plt.plot(EPISODES, REWARDS)
# plt.show()