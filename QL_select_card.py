import tensorflow as tf
from collections import deque
import numpy as np

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=225,#1086,
                 action_size=32, hidden_size1=10, hidden_size2=10,
                 hidden_size3=10, name='QNetworkCard'):
        """
        state_size: 1086 |225
            - state_overall:
                - game: 9
                - game_player: 4
                - first_player: 4
                - trick_nr: 8
                - course_of_game: 32*32=1024
                - davongelaufen: 1
                - scores: 4
            - state_player:
                - remaining_cards: 32
        action_size: 32 possible cards
        """
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputsCard')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actionsCard')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='targetCard')

            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size1)
            #self.bn1 = tf.contrib.layers.batch_norm(self.fc1)
            #self.do1 = tf.contrib.layers.dropout(self.fc1, keep_prob=0.7)

            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size2)
            #self.bn2 = tf.contrib.layers.batch_norm(self.fc2)
            #self.do2 = tf.contrib.layers.dropout(self.fc2, keep_prob=0.7)
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
