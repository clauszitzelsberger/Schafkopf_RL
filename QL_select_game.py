import tensorflow as tf
from collections import deque
import numpy as np

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=32,
                 action_size=9, hidden_size1=10, hidden_size2=10,
                 hidden_size3=10, name='QNetworkGame'):
        """
        state_size: 8 of 32 possible cards one hot encoded
        action_size: all games or no game
        """
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputsGame')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actionsGame')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='targetGame')

            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size1)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size2)
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size3)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc3, action_size,
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length 4, for four actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]
