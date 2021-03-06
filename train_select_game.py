from QL_select_game import QNetwork as QNetworkGame
from QL_select_game import Memory
from interface_to_states import interface_to_states
from rules import Rules
import helper_functions as h
import numpy as np
import random
import copy
import tensorflow as tf
import random

class train_select_game():
    def __init__(self):
        self.train_episodes = 3000          # max number of episodes to learn from
        self.gamma = 1                       # future reward discount

        # Exploration parameters
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.01            # minimum exploration probability 0.01
        self.decay_rate = 0.0001            # exponential decay rate for exploration prob

        # Network parameters
        self.hidden_size1 = 128               # number of units in each Q-network hidden layer 64
        self.hidden_size2 = 64
        self.hidden_size3 = 32
        self.learning_rate = 0.00001         # Q-network learning rate 0.00001

        # Memory parameters
        self.memory_size = 10000            # memory capacity
        self.batch_size = 64                # experience mini-batch size
        self.pretrain_length = self.batch_size   # number experiences to pretrain the memory

        tf.reset_default_graph()
        self.QNetworkGame = QNetworkGame(name='main',
                                         hidden_size1=self.hidden_size1,
                                         hidden_size2=self.hidden_size2,
                                         hidden_size3=self.hidden_size3,
                                         learning_rate=self.learning_rate)

        self.memory = Memory(max_size=self.memory_size)

        self.s = interface_to_states()

        self.rules = Rules()

        self.reward_scale = 210

    def populate_memory(self):

        # Make random actions and store experiences
        for _ in range(self.pretrain_length):
            self.s.reset_epsiode()
            self.s.dealing()

            for i in range(4):
                possible_games = self.s.return_possible_games(i)
                if len(possible_games) > 0:
                    selected_game = random.choice(possible_games)

                    if selected_game != [None, None]:
                        self.s.write_game_to_states(selected_game, i)
                
            # Action
            game = self.s.return_state_overall()['game']
            action = self.rules.get_index(game, 'game')

            # State
            if game == [None, None]:
                game_player = 0
            else:
                game_player = self.s.return_state_overall()['game_player']
            dealed_cards = self.s.return_state_player(game_player)['dealed_cards']
            dealed_cards_indexed = [self.rules.get_index(card, 'card') for card in dealed_cards]
            state = self.rules.get_one_hot_cards(dealed_cards_indexed)

            # Simulate playing
            if game != [None, None]:
                while self.s.return_state_overall()['trick_number'] < 8:
                    for i in range(4):
                        possible_cards = self.s.return_possbile_cards(i)
                        selected_card = random.choice(possible_cards)
                        self.s.write_card_to_states(selected_card, i)

                    self.s.update_first_player_trick_nr_score()

                rewards = self.s.return_reward_list()
                # Scale rewards
                rewards = [r/self.reward_scale for r in rewards]
            else:
                rewards = [0,0,0,0]

            # Reward
            reward = rewards[game_player]
            self.memory.add((state, action, reward))
        #print(self.memory.sample(20))


    def training(self):

        saver = tf.train.Saver()

        reward_list1 = []
        reward_list2 = []
        reward_list3 = []
        reward_list4 = []
        loss_list= []

        total_reward1=0
        total_reward2=0
        total_reward3=0
        total_reward4=0
        total_loss=0
        
        with tf.Session() as sess:

            # Initialize variables
            sess.run(tf.global_variables_initializer())

            for e in range(1, self.train_episodes+1):
                self.s.reset_epsiode()
                self.s.dealing()

                for i in range(4):
                    first_player = self.s.return_state_overall()['first_player']
                    possible_games = self.s.return_possible_games(i)
                    if len(possible_games) > 0:
                        # Explore or Exploit
                        explore_p = self.explore_stop + \
                            (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*1000*e)
                        if explore_p > np.random.rand() or \
                        self.s.return_state_player((first_player+i)%4)['player_id']!=0:
                            selected_game = random.choice(possible_games)

                        else:
                            dealed_cards = self.s.return_state_player((first_player+i)%4)['dealed_cards']
                            dealed_cards_indexed = [self.rules.get_index(card, 'card') for card in dealed_cards]
                            state = self.rules.get_one_hot_cards(dealed_cards_indexed)
                            state = np.array(state)
                            feed = {self.QNetworkGame.inputs_: state.reshape((1, *state.shape))}
                            Qs = sess.run(self.QNetworkGame.output, feed_dict=feed)
                            Qs = Qs[0].tolist()
                            possible_actions = [self.rules.get_index(p_g, 'game') for p_g in possible_games]
                            Qs_subset = [i for i in Qs if Qs.index(i) in possible_actions]
                            action = np.argmax(Qs_subset)
                            action = Qs.index(max(Qs_subset))
                            selected_game = self.rules.games[action]

                        if selected_game != [None, None]:
                                self.s.write_game_to_states(selected_game, i)


                # Action
                game = self.s.return_state_overall()['game']
                action = self.rules.get_index(game, 'game')

                # State
                if game == [None, None]:
                    game_player = 0
                else:
                    game_player = self.s.return_state_overall()['game_player']
                dealed_cards = self.s.return_state_player(game_player)['dealed_cards']
                dealed_cards_indexed = [self.rules.get_index(card, 'card') for card in dealed_cards]
                state = self.rules.get_one_hot_cards(dealed_cards_indexed)

                # Simulate playing
                if game != [None, None]:
                    while self.s.return_state_overall()['trick_number'] < 8:
                        for i in range(4):
                            possible_cards = self.s.return_possbile_cards(i)
                            selected_card = random.choice(possible_cards)
                            self.s.write_card_to_states(selected_card, i)

                        self.s.update_first_player_trick_nr_score()

                    rewards = self.s.return_reward_list()
                    # Scale rewards
                    rewards = [r/self.reward_scale for r in rewards]
                else:
                    rewards = [0,0,0,0]

                #print('States: {}'.format(self.s.return_state_overall()))
                #print('Rewards: {}'.format(rewards))

                # Reward
                reward = rewards[game_player]

                if game_player == 0:
                    self.memory.add((state, action, reward))

                reward1 = rewards[0]*self.reward_scale
                reward2 = rewards[1]*self.reward_scale
                reward3 = rewards[2]*self.reward_scale
                reward4 = rewards[3]*self.reward_scale

                # Sample mini-batch from memory
                batch = self.memory.sample(self.batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])

                # Train network
                target_Qs = sess.run(self.QNetworkGame.output,
                                    feed_dict={self.QNetworkGame.inputs_: states})

                targets = rewards + self.gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([self.QNetworkGame.loss, self.QNetworkGame.opt],
                                    feed_dict={self.QNetworkGame.inputs_: states,
                                               self.QNetworkGame.targetQs_: targets,
                                               self.QNetworkGame.actions_: actions})


                total_reward1+=reward1
                total_reward2+=reward2
                total_reward3+=reward3
                total_reward4+=reward4
                total_loss+=loss

                show_every = 100
                if e%show_every==0:
                    print('Episode: {}'.format(e),
                          'Total reward: {}'.format(reward1),
                          'Training loss: {:.4f}'.format(loss))
                    reward_list1.append(total_reward1/show_every)
                    reward_list2.append(total_reward2/show_every)
                    reward_list3.append(total_reward3/show_every)
                    reward_list4.append(total_reward4/show_every)
                    loss_list.append(total_loss/show_every)

                    total_reward1=0
                    total_reward2=0
                    total_reward3=0
                    total_reward4=0
                    total_loss=0

                if e%1000==0:
                    # Plot reward ~ epochs
                    h.plot_reward(reward_list1,
                             reward_list2,
                             reward_list3,
                             reward_list4,
                             show_every)

            # Plot loss ~ epochs
            h.plot_loss(loss_list, 'Select Game')

            # Save weights of NN
            saver.save(sess, "checkpoints/schafkopf.ckpt")

    
