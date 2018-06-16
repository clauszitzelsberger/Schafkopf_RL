from rules import Rules
from state_overall import state_overall
from state_player import state_player
from QL_choose_game import QNetwork
from QL_choose_game import Memory
import numpy as np
import random
import copy
import tensorflow as tf
import matplotlib.pyplot as plt

class playing_schafkopf():
    def __init__(self):
        self.rules = Rules()

        self.train_episodes = 10000          # max number of episodes to learn from
        #self.max_steps = 200                # max steps in an episode
        self.gamma = 1                       # future reward discount

        # Exploration parameters
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.01            # minimum exploration probability
        self.decay_rate = 0.0001            # exponential decay rate for exploration prob

        # Network parameters
        self.hidden_size = 64               # number of units in each Q-network hidden layer
        self.learning_rate = 0.00001         # Q-network learning rate

        # Memory parameters
        self.memory_size = 10000            # memory capacity
        self.batch_size = 20                # experience mini-batch size
        self.pretrain_length = self.batch_size   # number experiences to pretrain the memory

        tf.reset_default_graph()
        self.QNetwork = QNetwork(name='main',
                                 hidden_size=self.hidden_size,
                                 learning_rate=self.learning_rate)

        self.memory = Memory(max_size=self.memory_size)

    def reset_epsiode(self):
        player0 = state_player()
        player1 = state_player()
        player2 = state_player()
        player3 = state_player()

        player0.state_player['player_id']=0
        player1.state_player['player_id']=1
        player2.state_player['player_id']=2
        player3.state_player['player_id']=3

        self.players = [player0,
                        player1,
                        player2,
                        player3]

        self.state_overall = state_overall()

    def dealing(self):
        dealed_cards = self.rules.deal_cards()
        #print(dealed_cards)
        for i in range(len(self.players)):
            self.players[i].state_player['dealed_cards']=dealed_cards[i]
            self.players[i].state_player['remaining_cards']=dealed_cards[i]

    def new_game_to_choose(self, random_selection=True, sess=None):
        """
        random_selection: True, game selected radom,
            False, game selected by Q-Net
        """

        first_player = self.state_overall.state_overall['first_player']

        # order players
        players = [self.players[first_player],
                   self.players[(first_player+1)%4],
                   self.players[(first_player+2)%4],
                   self.players[(first_player+3)%4]]

        for i in range(len(players)):
            possible_games = players[i].\
                get_possible_games_to_play(self.state_overall)

            if random_selection:
                selected_game = random.choice(possible_games)
            else:
                dealed_cards = players[i].state_player['dealed_cards']
                state = [self.rules.get_index(card, 'card') for card in dealed_cards]
                state = np.array(state)
                feed = {self.QNetwork.inputs_: state.reshape((1, *state.shape))}
                Qs = sess.run(self.QNetwork.output, feed_dict=feed)
                Qs = Qs[0].tolist()
                possible_actions = [self.rules.get_index(p_g, 'game') for p_g in possible_games]
                Qs_subset = [i for i in Qs if Qs.index(i) in possible_actions]
                action = np.argmax(Qs_subset)
                selected_game = self.rules.games[action]

            if selected_game != [None, None]:
                self.state_overall.state_overall['game']=selected_game
                self.state_overall.state_overall['game_player']=(first_player+i)%4
                break

    def play(self):

        trick_nr = self.state_overall.state_overall['trick_number']

        # order of players
        first_player = self.state_overall.state_overall['first_player']

        players = [self.players[first_player],
                   self.players[(first_player+1)%4],
                   self.players[(first_player+2)%4],
                   self.players[(first_player+3)%4]]

        for i in range(len(players)):
            possible_cards = players[i].\
                get_possible_cards_to_play(self.state_overall)
            selected_card = random.choice(possible_cards)
            #print('Possible Cards: {}'.format(possible_cards))
            #print('Selected Cards: {}'.format(selected_card))

            # Update states for next player
            remaining_cards=copy.copy(players[i].state_player['remaining_cards'])
            remaining_cards.remove(selected_card)
            #remaining_cards.append([None, None])
            players[i].state_player['remaining_cards'] = remaining_cards
            #players[i].state_player['remaining_cards'].append([None, None])
            self.state_overall.state_overall['course_of_game'][trick_nr][(first_player+i)%4] = \
                selected_card
            # Not implemented yet: logic missing if davongelaufen True or False
            self.state_overall.state_overall['davongelaufen'] = False

        # Update first_player and trick_number
        highest_card = self.state_overall.highest_card_of_played_cards()
        self.state_overall.state_overall['first_player'] = \
            highest_card
        self.state_overall.state_overall['scores'][highest_card] += \
            self.state_overall.get_score()
        self.state_overall.state_overall['trick_number'] += 1

    def populate_memory(self):


        # Make random actions and store experiences
        for i in range(self.pretrain_length):
            self.reset_epsiode()
            self.dealing()
            self.new_game_to_choose()

            # Action
            game = self.state_overall.state_overall['game']
            action = self.rules.get_index(game, 'game')

            # State
            if game == [None, None]:
                game_player = 0
            else:
                game_player = self.state_overall.state_overall['game_player']
            dealed_cards = self.players[game_player].state_player['dealed_cards']
            state = [self.rules.get_index(card, 'card') for card in dealed_cards]

            if game != [None, None]:
                while self.state_overall.state_overall['trick_number'] < 8:
                    self.play()
                rewards = self.state_overall.get_reward_list()
            else:
                rewards = [0,0,0,0]

            # Reward
            reward = rewards[game_player]
            self.memory.add((state, action, reward))
        #print(self.memory.sample(self.pretrain_length))


    def training(self):

        saver = tf.train.Saver()
        reward_list_100 = []
        total_reward=0
        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())

            for e in range(1, self.train_episodes):
                self.reset_epsiode()
                self.dealing()
                #total_reward = 0
                #t = 0

                # Explore or Exploit
                explore_p = self.explore_stop + \
                    (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*1000*e)
                if explore_p > np.random.rand():
                    # Make a random action
                    self.new_game_to_choose()
                else:
                    # Get action from Q-network
                    self.new_game_to_choose(False, sess)

                # Action
                game = self.state_overall.state_overall['game']
                action = self.rules.get_index(game, 'game')

                # State
                if game == [None, None]:
                    game_player = 0
                else:
                    game_player = self.state_overall.state_overall['game_player']
                dealed_cards = self.players[game_player].state_player['dealed_cards']
                state = [self.rules.get_index(card, 'card') for card in dealed_cards]

                if self.state_overall.state_overall['game'] != [None, None]:
                    while self.state_overall.state_overall['trick_number'] < 8:
                        self.play()
                    rewards = self.state_overall.get_reward_list()
                else:
                    rewards = [0,0,0,0]

                # Reward
                reward = rewards[game_player]
                self.memory.add((state, action, reward))




                # Sample mini-batch from memory
                batch = self.memory.sample(self.batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])

                # Train network
                target_Qs = sess.run(self.QNetwork.output,
                                    feed_dict={self.QNetwork.inputs_: states})

                targets = rewards + self.gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([self.QNetwork.loss, self.QNetwork.opt],
                                    feed_dict={self.QNetwork.inputs_: states,
                                               self.QNetwork.targetQs_: targets,
                                               self.QNetwork.actions_: actions})

                total_reward+=reward
                if e%100==0:
                    print('Episode: {}'.format(e),
                          'Total reward: {}'.format(reward),
                          'Training loss: {:.4f}'.format(loss))
                    reward_list_100.append(total_reward)
                    total_reward=0



            #print('Total Reward: {}'.format(total_reward))
            self.plot_reward(reward_list_100)


            saver.save(sess, "checkpoints/schafkopf.ckpt")

    def plot_reward(self, reward_list):
        x = range(len(reward_list))
        y = reward_list
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='100 epochs', ylabel='total reward in 100 epochs',
               title='Reward ~ epochs')
        plt.show()
