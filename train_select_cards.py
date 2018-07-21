from QL_select_card import QNetwork as QNetworkCard
from QL_select_card import Memory
from interface_to_states import interface_to_states
from rules import Rules
import helper_functions as h
import numpy as np
import random
import copy
import tensorflow as tf
import random
import math

class train_select_cards():
    def __init__(self):
        self.train_episodes = 2000          # max number of episodes to learn from
        self.gamma = 1                       # future reward discount

        # Exploration parameters
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.01            # minimum exploration probability 0.01
        self.decay_rate = 0.0001            # exponential decay rate for exploration prob 0.00001

        # Network parameters
        self.hidden_size1 = 64               # number of units in each Q-network hidden layer 64
        self.hidden_size2 = 32
        self.hidden_size3 = 16
        self.learning_rate = 0.00001        # Q-network learning rate 0.00001

        # Memory parameters
        self.memory_size = 1000             # memory capacity
        self.batch_size = 8                # experience mini-batch size
        self.pretrain_length = self.batch_size*8   # number experiences to pretrain the memory

        tf.reset_default_graph()
        self.QNetworkCard = QNetworkCard(name='main',
                                         hidden_size1=self.hidden_size1,
                                         hidden_size2=self.hidden_size2,
                                         hidden_size3=self.hidden_size3,
                                         learning_rate=self.learning_rate)

        self.memory = Memory(max_size=self.memory_size)

        self.s = interface_to_states()

        self.rules = Rules()

        self.reward_scale = 210 # lost solo schneider schwarz

    def populate_memory(self):

        # Make random actions and store experiences
        j = 0
        while j < math.ceil(self.pretrain_length/8):
            self.s.reset_epsiode()
            self.s.dealing()

            for i in range(4):
                possible_games = self.s.return_possible_games(i)
                if len(possible_games) > 0:
                    selected_game = random.choice(possible_games)

                    if selected_game != [None, None]:
                        self.s.write_game_to_states(selected_game, i)
            
            # Simulate playing
            if self.s.return_state_overall()['game'] != [None, None]:
                states_list = []
                action_list = []
                while self.s.return_state_overall()['trick_number'] < 8:
                    j+=1
                    first_player = self.s.return_state_overall()['first_player']
                    for i in range(4):
                        possible_cards = self.s.return_possbile_cards(i)
                        selected_card = random.choice(possible_cards)

                        # Save player 0's actions and states in lists
                        if ((first_player+i)%4) == 0:
                            state = self.s.return_state_select_card(player_id=0)
                            action = self.rules.get_index(selected_card, 'card')
                            states_list.append(state)
                            action_list.append(action)
                            
                        # Write cards to states
                        self.s.write_card_to_states(selected_card, i)
                    
                    # Update states
                    self.s.update_first_player_trick_nr_score()
                    
                rewards = self.s.return_reward_list()
                # Scale rewards
                rewards = [r/self.reward_scale for r in rewards]

                # Save action, states and reward in memory
                for i in range(len(states_list)):
                    if i < 7:
                        self.memory.add((states_list[i],
                                    action_list[i],
                                    0, #reward
                                    states_list[i+1]))
                    else:
                        self.memory.add((states_list[i],
                                    action_list[i],
                                    rewards[0],
                                    np.zeros(np.array(state).shape).tolist()))


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
                    possible_games = self.s.return_possible_games(i)

                    if len(possible_games) > 0:
                        selected_game = random.choice(possible_games)

                        if selected_game != [None, None]:
                            self.s.write_game_to_states(selected_game, i)
            
                # Simulate playing
                if self.s.return_state_overall()['game'] != [None, None]:
                    states_list = []
                    action_list = []
                    while self.s.return_state_overall()['trick_number'] < 8:
                        first_player = self.s.return_state_overall()['first_player']
                        for i in range(4):
                            possible_cards = self.s.return_possbile_cards(i)

                            # Explore or Exploit
                            explore_p = self.explore_stop + \
                                (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*1*e)
                            if explore_p > np.random.rand() or \
                            self.s.return_state_player((first_player+i)%4)['player_id']!=0:
                                selected_card = random.choice(possible_cards)

                                if self.s.return_state_player((first_player+i)%4)['player_id']==0:
                                    state = self.s.return_state_select_card(player_id=0)
                                    state = np.array(state)
                                    action = self.rules.get_index(selected_card, 'card')
                                    states_list.append(state)
                                    action_list.append(action)

                            else:
                                state = self.s.return_state_select_card(player_id=0)
                                state = np.array(state)
                                feed = {self.QNetworkCard.inputs_: state.reshape((1, *state.shape))}
                                Qs = sess.run(self.QNetworkCard.output, feed_dict=feed)
                                Qs = Qs[0].tolist()
                                possible_actions = [self.rules.get_index(p_g, 'card') for p_g in possible_cards]
                                Qs_subset = [i for i in Qs if Qs.index(i) in possible_actions]
                                action = np.argmax(Qs_subset)
                                action = Qs.index(max(Qs_subset))
                                selected_card = self.rules.cards[action]
                                states_list.append(state)
                                action_list.append(action)

                            # Write cards to states
                            self.s.write_card_to_states(selected_card, i)

                        # Update states
                        self.s.update_first_player_trick_nr_score()

                    rewards = self.s.return_reward_list()
                    # Scale rewards
                    rewards = [r/self.reward_scale for r in rewards]

                    reward1 = rewards[0]*self.reward_scale
                    reward2 = rewards[1]*self.reward_scale
                    reward3 = rewards[2]*self.reward_scale
                    reward4 = rewards[3]*self.reward_scale


                    # Save action, states and reward in memory
                    for i in range(len(states_list)):
                        if i < 7:
                            self.memory.add((states_list[i],
                                        action_list[i],
                                        0, #reward
                                        states_list[i+1]))
                        else:
                            self.memory.add((states_list[i],
                                        action_list[i],
                                        rewards[0],
                                        np.zeros(np.array(state).shape).tolist()))


                    # Sample mini-batch from memory
                    batch = self.memory.sample(self.batch_size)
                    states = np.array([each[0] for each in batch])
                    actions = np.array([each[1] for each in batch])
                    rewards = np.array([each[2] for each in batch])
                    next_states = np.array([each[3] for each in batch])

                    # Train network
                    target_Qs = sess.run(self.QNetworkCard.output,
                                        feed_dict={self.QNetworkCard.inputs_: next_states}) #states})

                    targets = rewards + self.gamma * np.max(target_Qs, axis=1)

                    loss, _ = sess.run([self.QNetworkCard.loss, self.QNetworkCard.opt],
                                        feed_dict={self.QNetworkCard.inputs_: states,
                                                    self.QNetworkCard.targetQs_: targets,
                                                    self.QNetworkCard.actions_: actions})

                        


                    total_reward1+=reward1
                    total_reward2+=reward2
                    total_reward3+=reward3
                    total_reward4+=reward4
                    total_loss+=loss
                    
                else:
                    rewards = [0,0,0,0]

                

                show_every = 500
                if e%show_every==0:
                    print('Episode: {}'.format(e),
                          'Avg. total reward: {:.1f}'.format(total_reward1/show_every),
                          'Avg. training loss: {:.5f}'.format(total_loss/show_every))
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
                             reward_list4, show_every)

            # Plot loss ~ epochs
            h.plot_loss(loss_list)

            # Save weights of NN
            saver.save(sess, "checkpoints/schafkopf.ckpt")

            # Print average reward
            print('Total reward: \n')
            print('0: {:.1f}, \n1: {:.1f}, \n2: {:.1f}, \n3: {:.1f}'.format(sum(reward_list1),
                                                                            sum(reward_list2),
                                                                            sum(reward_list3),
                                                                            sum(reward_list4)))
