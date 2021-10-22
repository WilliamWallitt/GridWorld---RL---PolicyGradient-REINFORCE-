import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow_probability.python.distributions as dist


class Model:

    def __init__(self):
        self.model = keras.models.Sequential()
        self.states_memory = []
        self.rewards_memory = []
        self.action_memory = []

    def create_model(self, map_size, feature_planes, output_space):
        self.model.add(keras.layers.Input(shape=(map_size, map_size, feature_planes)))
        self.model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(64, activation='relu'))
        self.model.add(keras.layers.Dense(output_space, activation='softmax'))
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01))

    def train_model(self):

        G = np.zeros_like(self.rewards_memory)
        for t in range(len(self.rewards_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.rewards_memory)):
                G_sum += self.rewards_memory[k] * discount
                discount *= 0.90
            G[t] = G_sum

        G = (G - np.mean(G)) / (np.std(G) + 1e-7)
        actions = tf.convert_to_tensor(self.action_memory)

        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.states_memory)):
                # g is reward
                state = tf.convert_to_tensor([state])
                model_probabilities = self.model(state)
                action_probabilities = dist.Categorical(probs=model_probabilities)
                log_probabilities = action_probabilities.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_probabilities)

        # get the current networks gradient with respect to the loss
        gradient = tape.gradient(loss, self.model.trainable_variables)
        # adjust the network in the direction of the gradient
        self.model.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

    # not being used atm
    @staticmethod
    def discounted_rewards(rewards, gamma=0.8):
        discounted_rewards = []
        discounted_sum = 0
        for r in rewards[::-1]:
            discounted_sum = r + gamma * discounted_sum
            discounted_rewards.insert(0, discounted_sum)
        return discounted_rewards

