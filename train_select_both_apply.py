
from QL_select_card import QNetwork as QNetworkCard
from QL_select_game import QNetwork as QNetworkGame
from QL_select_card import Memory as MemoryCard
from QL_select_game import Memory as MemoryGame
from interface_to_states import interface_to_states
from rules import Rules
import helper_functions as h
import numpy as np
import random
import copy
import tensorflow as tf
import random
import math
from QL_select_game import QNetwork as QNetworkGame
from QL_select_game import Memory


class train_select_both_apply():
    def __init__(self):
        self.show_plots = False
        self.show_course_of_game = True
        self.train_episodes = 10000          # max number of episodes to learn from
        self.gamma = 1                       # future reward discount

        # Exploration parameters
        self.explore_startC = 0.8            # exploration probability at start
        self.explore_stopC = 0.05            # minimum exploration probability 0.01
        self.decay_rateC = 0.0005            # exponential decay rate for exploration prob 0.00001

         # Exploration parameters
        self.explore_startG = 0.8            # exploration probability at start
        self.explore_stopG = 0.05            # minimum exploration probability 0.01
        self.decay_rateG = 0.001            # exponential decay rate for exploration prob


        # Network parameters
        self.hidden_size1C = 64               # number of units in each Q-network hidden layer 64
        self.hidden_size2C = 64
        self.hidden_size3C = 64
        self.learning_rateC = 0.000001        # Q-network learning rate 0.00001

         # Network parameters
        self.hidden_size1G = 64               # number of units in each Q-network hidden layer 64
        self.hidden_size2G = 64
        self.hidden_size3G = 64
        self.learning_rateG = 0.00001         # Q-network learning rate 0.00001



        # Memory parameters
        self.memory_sizeC = 1000             # memory capacity
        self.batch_sizeC = 64                # experience mini-batch size
        self.pretrain_lengthC = self.batch_sizeC*8   # number experiences to pretrain the memory

        # Memory parameters
        self.memory_sizeG = 1000            # memory capacity
        self.batch_sizeG = 64                # experience mini-batch size
        self.pretrain_lengthG = self.batch_sizeG   # number experiences to pretrain the memory



        tf.reset_default_graph()
        self.QNetworkCard = QNetworkCard(name='mainC',
                                         hidden_size1=self.hidden_size1C,
                                         hidden_size2=self.hidden_size2C,
                                         hidden_size3=self.hidden_size3C,
                                         learning_rate=self.learning_rateC)

        self.TargetNetworkCard = QNetworkCard(name='targetNetworkC',
                                              hidden_size1=self.hidden_size1C,
                                              hidden_size2=self.hidden_size2C,
                                              hidden_size3=self.hidden_size3C,
                                              learning_rate=self.learning_rateC)

        self.QNetworkGame = QNetworkGame(name='mainG',
                                         hidden_size1=self.hidden_size1G,
                                         hidden_size2=self.hidden_size2G,
                                         hidden_size3=self.hidden_size3G,
                                         learning_rate=self.learning_rateG)

        self.TargetNetworkGame = QNetworkGame(name='targetNetworkG',
                                         hidden_size1=self.hidden_size1G,
                                         hidden_size2=self.hidden_size2G,
                                         hidden_size3=self.hidden_size3G,
                                         learning_rate=self.learning_rateG)

        self.memoryGame = Memory(max_size=self.memory_sizeG)

        self.memoryCard = MemoryCard(max_size=self.memory_sizeC)

        self.s = interface_to_states()

        self.rules = Rules()

        self.reward_scale = 210 # lost solo schneider schwarz
        #self.score_scale = 120

        self.max_tau = 1000 # update Target Network with the DQN mainC/mainG

    def populate_memoryC(self):

        # Make random actions and store experiences
        j = 0
        while j < math.ceil(self.pretrain_lengthC/8):
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
                #score_list = []
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

                    # Old score
                    #old_score=self.s.return_state_overall()['scores'][0]

                    # Update states
                    self.s.update_first_player_trick_nr_score()

                    # New score
                    #new_score=self.s.return_state_overall()['scores'][0]

                    #score_list.append(new_score-old_score)

                rewards = self.s.return_reward_list()
                # Scale rewards
                rewards = [r/self.reward_scale for r in rewards]
                #scores = [s/self.score_scale for s in score_list]


                # Save action, states and reward in memory
                for i in range(len(states_list)):
                    if i < 7:
                        self.memoryCard.add((states_list[i],
                                    action_list[i],
                                    rewards[0],#0, #reward#scores[i],#
                                    states_list[i+1]))
                    else:
                        self.memoryCard.add((states_list[i],
                                    action_list[i],
                                    rewards[0],#scores[i],#
                                    np.zeros(np.array(state).shape).tolist()))

    def populate_memoryG(self):

        # Make random actions and store experiences
        for _ in range(self.pretrain_lengthG):
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
            self.memoryGame.add((state, action, reward, np.zeros(np.array(state).shape).tolist()))

    def update_target_graph(self, model_type='C'):

        DQN_model = 'main' + model_type
        TargetNetwork_model = 'targetNetwork' + model_type
    
        # Get the parameters of our DQNNetwork
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, DQN_model)
    
        # Get the parameters of our Target_network
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, TargetNetwork_model)

        op_holder = []
    
        # Update our target_network parameters with DQNNetwork parameters
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def training(self):

        saver = tf.train.Saver()

        reward_list1 = []
        reward_list2 = []
        reward_list3 = []
        reward_list4 = []
        loss_listG= []
        loss_listC=[]

        total_reward1=0
        total_reward2=0
        total_reward3=0
        total_reward4=0
        total_lossG=0
        total_lossC=0
        tau=0

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
                        explore_pG = self.explore_stopG + \
                            (self.explore_startG - self.explore_stopG)*np.exp(-self.decay_rateG*1*e)
                        if explore_pG > np.random.rand(): #or \
                        #self.s.return_state_player((first_player+i)%4)['player_id']!=0:
                            selected_game = random.choice(possible_games)

                        else:
                            dealed_cards = self.s.return_state_player((first_player+i)%4)['dealed_cards']
                            dealed_cards_indexed = [self.rules.get_index(card, 'card') for card in dealed_cards]
                            stateG = self.rules.get_one_hot_cards(dealed_cards_indexed)
                            stateG = np.array(stateG)

                            # Player 0 acts with constantly updating Network, other players act with old network which
                            # will be updated every x episodes
                            if self.s.return_state_player((first_player+i)%4)['player_id']!=0:# and e<3000:
                                feedG = {self.TargetNetworkGame.inputs_: stateG.reshape((1, *stateG.shape))}
                                QsG = sess.run(self.TargetNetworkGame.output, feed_dict=feedG)
                            else:
                                feedG = {self.QNetworkGame.inputs_: stateG.reshape((1, *stateG.shape))}
                                QsG = sess.run(self.QNetworkGame.output, feed_dict=feedG)

                            QsG = QsG[0].tolist()
                            possible_actionsG = [self.rules.get_index(p_g, 'game') for p_g in possible_games]
                            Qs_subsetG = [i for i in QsG if QsG.index(i) in possible_actionsG]
                            actionG = np.argmax(Qs_subsetG)
                            actionG = QsG.index(max(Qs_subsetG))
                            selected_game = self.rules.games[actionG]

                        if selected_game != [None, None]:
                                self.s.write_game_to_states(selected_game, i)


                # Action
                game = self.s.return_state_overall()['game']
                actionG = self.rules.get_index(game, 'game')

                # State
                if game == [None, None]:
                    game_player = random.choice([0,1,2,3])
                else:
                    game_player = self.s.return_state_overall()['game_player']
                dealed_cards = self.s.return_state_player(game_player)['dealed_cards']
                dealed_cards_indexed = [self.rules.get_index(card, 'card') for card in dealed_cards]
                stateG = self.rules.get_one_hot_cards(dealed_cards_indexed)




                # Simulate playing
                if self.s.return_state_overall()['game'] != [None, None]:

                    # Show cards of player 0 when he is player
                    if e > 5000:
                        if self.s.return_state_overall()['game_player']==0:
                            if self.show_course_of_game:
                                print(self.rules.name_of_cards(dealed_cards))

                    states_listC = []
                    action_listC = []
                    #score_list = []
                    while self.s.return_state_overall()['trick_number'] < 8:
                        first_player = self.s.return_state_overall()['first_player']
                        for i in range(4):
                            possible_cards = self.s.return_possbile_cards(i)

                            # Explore or Exploit
                            explore_pC = self.explore_stopC + \
                                (self.explore_startC - self.explore_stopC)*np.exp(-self.decay_rateC*1*e)
                            if explore_pC > np.random.rand():# or \
                            #self.s.return_state_player((first_player+i)%4)['player_id']!=0:
                                selected_card = random.choice(possible_cards)

                                if self.s.return_state_player((first_player+i)%4)['player_id']==0:
                                    stateC = self.s.return_state_select_card(player_id=0)
                                    stateC = np.array(stateC)
                                    actionC = self.rules.get_index(selected_card, 'card')
                                    states_listC.append(stateC)
                                    action_listC.append(actionC)

                            else:
                                stateC = self.s.return_state_select_card(player_id=0)
                                stateC = np.array(stateC)

                                # Player 0 acts with constantly updating Network, other players act with old network which
                                # will be updated every x episodes
                                if self.s.return_state_player((first_player+i)%4)['player_id']!=0:# and e<3000:
                                    feedC = {self.TargetNetworkCard.inputs_: stateC.reshape((1, *stateC.shape))}
                                    QsC = sess.run(self.TargetNetworkCard.output, feed_dict=feedC)
                                else:
                                    feedC = {self.QNetworkCard.inputs_: stateC.reshape((1, *stateC.shape))}
                                    QsC = sess.run(self.QNetworkCard.output, feed_dict=feedC)

                                feedC = {self.QNetworkCard.inputs_: stateC.reshape((1, *stateC.shape))}
                                QsC = sess.run(self.QNetworkCard.output, feed_dict=feedC)
                                QsC = QsC[0].tolist()
                                possible_actionsC = [self.rules.get_index(p_g, 'card') for p_g in possible_cards]
                                Qs_subsetC = [i for i in QsC if QsC.index(i) in possible_actionsC]
                                actionC = np.argmax(Qs_subsetC)
                                actionC = QsC.index(max(Qs_subsetC))
                                selected_card = self.rules.cards[actionC]

                                if self.s.return_state_player((first_player+i)%4)['player_id']==0:
                                    states_listC.append(stateC)
                                    action_listC.append(actionC)

                            # Write cards to states
                            self.s.write_card_to_states(selected_card, i)

                            # Visualize course of game
                            if e > 5000:
                                if self.s.return_state_overall()['game_player']==0:
                                    h.return_course_of_game(self.s.return_state_overall(), self.show_course_of_game)


                        # Old score
                        #old_score=self.s.return_state_overall()['scores'][0]

                        # Update states
                        self.s.update_first_player_trick_nr_score()

                        # New score
                        #new_score=self.s.return_state_overall()['scores'][0]

                        #score_list.append(new_score-old_score)

                    rewards = self.s.return_reward_list()

                    # Scale rewards and scores
                    # Three possibilities: either use reward at the end of the game or score delta
                    # after very tick as reward for selecting cards -> second one makes no sense as scores
                    # of parter are not rewarded
                    # 3. give reward to every step
                    rewards = [r/self.reward_scale for r in rewards]
                    #scores = [s/self.score_scale for s in score_list]

                    reward1 = rewards[0]*self.reward_scale
                    reward2 = rewards[1]*self.reward_scale
                    reward3 = rewards[2]*self.reward_scale
                    reward4 = rewards[3]*self.reward_scale


                    # Save action, states and reward in memory
                    for i in range(len(states_listC)):
                        if i < 7:
                            self.memoryCard.add((states_listC[i],
                                        action_listC[i],
                                        rewards[0],#0, #reward#scores[i],#
                                        states_listC[i+1]))
                        else:
                            self.memoryCard.add((states_listC[i],
                                        action_listC[i],
                                        rewards[0],#scores[i],#
                                        np.zeros(np.array(stateC).shape).tolist()))

                    




                    # Sample mini-batch from memory Card
                    batchC = self.memoryCard.sample(self.batch_sizeC)
                    statesC = np.array([each[0] for each in batchC])
                    actionsC = np.array([each[1] for each in batchC])
                    rewardsC = np.array([each[2] for each in batchC])
                    next_statesC = np.array([each[3] for each in batchC])

                    # Train network
                    target_QsC = sess.run(self.QNetworkCard.output,
                                        feed_dict={self.QNetworkCard.inputs_: next_statesC}) 

                    # Set target_Qs to 0 for states where episode ends
                    episode_endsC = (next_statesC == np.zeros(statesC[0].shape)).all(axis=1)
                    target_QsC[episode_endsC] = np.zeros(target_QsC.shape[1])
                    
                    targetsC = rewardsC + self.gamma * np.max(target_QsC, axis=1)

                    lossC, _ = sess.run([self.QNetworkCard.loss, self.QNetworkCard.opt],
                                        feed_dict={self.QNetworkCard.inputs_: statesC,
                                                    self.QNetworkCard.targetQs_: targetsC,
                                                    self.QNetworkCard.actions_: actionsC})


                    



                    total_reward1+=reward1
                    total_reward2+=reward2
                    total_reward3+=reward3
                    total_reward4+=reward4
                    total_lossC+=lossC

                else:
                    rewards = [0,0,0,0]



                if game_player==0:
                    self.memoryGame.add((stateG, actionG, rewards[game_player], np.zeros(np.array(stateG).shape).tolist()))

                # Sample mini-batch from memory Game
                batchG = self.memoryGame.sample(self.batch_sizeG)
                statesG = np.array([each[0] for each in batchG])
                actionsG = np.array([each[1] for each in batchG])
                rewardsG = np.array([each[2] for each in batchG])
                next_statesG = np.array([each[3] for each in batchG])

                # Train network
                target_QsG = sess.run(self.QNetworkGame.output,
                                    feed_dict={self.QNetworkGame.inputs_: next_statesG})

                # Set target_Qs to 0 for states where episode ends
                #episode_endsG = (next_statesG == np.zeros(statesG[0].shape)).all(axis=1)
                #target_QsG[episode_endsG] = (0, 0)

                targetsG = rewardsG + self.gamma * np.max(target_QsG, axis=1)

                lossG, _ = sess.run([self.QNetworkGame.loss, self.QNetworkGame.opt],
                                    feed_dict={self.QNetworkGame.inputs_: statesG,
                                                self.QNetworkGame.targetQs_: targetsG,
                                                self.QNetworkGame.actions_: actionsG})
                total_lossG+=lossG



                show_every = 100

                if e%show_every==0:
                    print('Episode: {}'.format(e),
                          'Avg. total reward: {:.1f}'.format(total_reward1/show_every),
                          'Avg. training lossG: {:.5f}'.format(total_lossG/show_every),
                          'Avg. training lossC: {:.5f}'.format(total_lossC/show_every))
                    reward_list1.append(total_reward1/show_every)
                    reward_list2.append(total_reward2/show_every)
                    reward_list3.append(total_reward3/show_every)
                    reward_list4.append(total_reward4/show_every)
                    loss_listG.append(total_lossG/show_every)
                    loss_listC.append(total_lossC/show_every)

                    total_reward1=0
                    total_reward2=0
                    total_reward3=0
                    total_reward4=0
                    total_lossG=0
                    total_lossC=0


                if tau >= self.max_tau:
                        update_targetC = self.update_target_graph(model_type='C')
                        sess.run(update_targetC)
                        #print('ModelC updated')
                        update_targetG = self.update_target_graph(model_type='G')
                        sess.run(update_targetG)
                        #print('ModelG updated')
                        tau = 0

                tau+=1

                if e%500==0:
                    #Plot reward ~ epochs
                    h.plot_reward(reward_list1,
                             reward_list2,
                             reward_list3,
                             reward_list4, show_every, self.show_plots)

            # Plot loss ~ epochs
            h.plot_loss(loss_listG, 'Select Game', self.show_plots)
            h.plot_loss(loss_listC, 'Select Cards', self.show_plots)

            # Save weights of NN
            saver.save(sess, "checkpoints/schafkopf.ckpt")

            # Print average reward
            print('Avg. reward: \n')
            print('0: {:.1f}, \n1: {:.1f}, \n2: {:.1f}, \n3: {:.1f}'.format(np.mean(reward_list1),
                                                                            np.mean(reward_list2),
                                                                            np.mean(reward_list3),
                                                                            np.mean(reward_list4)))

    def apply_model(self):

        saver = tf.train.Saver()

        reward_list1 = []
        reward_list2 = []
        reward_list3 = []
        reward_list4 = []
        #loss_list= []

        total_reward1=0
        total_reward2=0
        total_reward3=0
        total_reward4=0
        #total_loss=0

        with tf.Session() as sess:

            # Initialize variables
            #sess.run(tf.global_variables_initializer())

            # Restore model
            saver = tf.train.import_meta_graph('C:/Users/claus/OneDrive/Documents/Schafkopf_RL/Schafkopf_RL-master/checkpoints/schafkopf.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('C:/Users/claus/OneDrive/Documents/Schafkopf_RL/Schafkopf_RL-master/checkpoints/'))

            for e in range(1, self.train_episodes+1):
                self.s.reset_epsiode()
                self.s.dealing()

                for i in range(4):
                    first_player = self.s.return_state_overall()['first_player']
                    possible_games = self.s.return_possible_games(i)

                    if len(possible_games) > 0:
                        #if self.s.return_state_player((first_player+i)%4)['player_id']!=0:
                        #    selected_game = random.choice(possible_games)
                        #else:
                        dealed_cards = self.s.return_state_player((first_player+i)%4)['dealed_cards']
                        dealed_cards_indexed = [self.rules.get_index(card, 'card') for card in dealed_cards]
                        stateG = self.rules.get_one_hot_cards(dealed_cards_indexed)
                        stateG = np.array(stateG)
                        feedG = {self.QNetworkGame.inputs_: stateG.reshape((1, *stateG.shape))}
                        QsG = sess.run(self.QNetworkGame.output, feed_dict=feedG)
                        QsG = QsG[0].tolist()
                        possible_actionsG = [self.rules.get_index(p_g, 'game') for p_g in possible_games]
                        Qs_subsetG = [i for i in QsG if QsG.index(i) in possible_actionsG]
                        actionG = np.argmax(Qs_subsetG)
                        actionG = QsG.index(max(Qs_subsetG))
                        selected_game = self.rules.games[actionG]

                        if selected_game != [None, None]:
                            self.s.write_game_to_states(selected_game, i)

                # Action
                game = self.s.return_state_overall()['game']
                actionG = self.rules.get_index(game, 'game')

                # State
                if game == [None, None]:
                    game_player = 0
                else:
                    game_player = self.s.return_state_overall()['game_player']
                dealed_cards = self.s.return_state_player(game_player)['dealed_cards']
                dealed_cards_indexed = [self.rules.get_index(card, 'card') for card in dealed_cards]
                stateG = self.rules.get_one_hot_cards(dealed_cards_indexed)


                # Simulate playing
                if self.s.return_state_overall()['game'] != [None, None]:

                    if self.s.return_state_overall()['game_player']==0:
                        if self.show_course_of_game:
                            print(self.rules.name_of_cards(dealed_cards))

                    states_list = []
                    action_list = []
                    while self.s.return_state_overall()['trick_number'] < 8:
                        first_player = self.s.return_state_overall()['first_player']
                        for i in range(4):
                            possible_cards = self.s.return_possbile_cards(i)

                            #if self.s.return_state_player((first_player+i)%4)['player_id']!=0:
                            #    selected_card = random.choice(possible_cards)
                            #
                            #    if self.s.return_state_player((first_player+i)%4)['player_id']==0:
                            #        state = self.s.return_state_select_card(player_id=0)
                            #        state = np.array(state)
                            #        action = self.rules.get_index(selected_card, 'card')
                            #        states_list.append(state)
                            #        action_list.append(action)

                            #else:
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

                            if self.s.return_state_player((first_player+i)%4)['player_id']==0:
                                states_list.append(state)
                                action_list.append(action)

                            # Write cards to states
                            self.s.write_card_to_states(selected_card, i)

                            # Print course of game
                            if self.s.return_state_overall()['game_player']==0:
                                h.return_course_of_game(self.s.return_state_overall(), self.show_course_of_game)


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
                            self.memoryCard.add((states_list[i],
                                        action_list[i],
                                        rewards[0], #reward
                                        states_list[i+1]))
                        else:
                            self.memoryCard.add((states_list[i],
                                        action_list[i],
                                        rewards[0],
                                        np.zeros(np.array(state).shape).tolist()))

                    if game_player==0:
                        self.memoryGame.add((stateG, actionG, rewards[0], np.zeros(np.array(stateG).shape).tolist()))

                    # Sample mini-batch from memory
                    #batch = self.memory.sample(self.batch_size)
                    #states = np.array([each[0] for each in batch])
                    #actions = np.array([each[1] for each in batch])
                    #rewards = np.array([each[2] for each in batch])
                    #next_states = np.array([each[3] for each in batch])

                    # Train network
                    #target_Qs = sess.run(self.QNetworkCard.output,
                    #                    feed_dict={self.QNetworkCard.inputs_: next_states}) #states})

                    #targets = rewards + self.gamma * np.max(target_Qs, axis=1)

                    #loss, _ = sess.run([self.QNetworkCard.loss, self.QNetworkCard.opt],
                    #                    feed_dict={self.QNetworkCard.inputs_: states,
                    #                                self.QNetworkCard.targetQs_: targets,
                    #                                self.QNetworkCard.actions_: actions})




                    total_reward1+=reward1
                    total_reward2+=reward2
                    total_reward3+=reward3
                    total_reward4+=reward4
                    #total_loss+=loss

                else:
                    rewards = [0,0,0,0]



                show_every = 500
                if e%show_every==0:
                    print('Episode: {}'.format(e),
                            'Avg. total reward: {:.1f}'.format(total_reward1/show_every))
                    reward_list1.append(total_reward1/show_every)
                    reward_list2.append(total_reward2/show_every)
                    reward_list3.append(total_reward3/show_every)
                    reward_list4.append(total_reward4/show_every)
                    #loss_list.append(total_loss/show_every)

                    total_reward1=0
                    total_reward2=0
                    total_reward3=0
                    total_reward4=0
                    #total_loss=0

                if e%1000==0:
                    # Plot reward ~ epochs
                    h.plot_reward(reward_list1,
                                reward_list2,
                                reward_list3,
                                reward_list4, show_every, self.show_plots)

            # Plot loss ~ epochs
            #h.plot_loss(loss_list)

            # Save weights of NN
            #saver.save(sess, "checkpoints/schafkopf.ckpt")

            # Print average reward
            print('Total reward: \n')
            print('0: {:.1f}, \n1: {:.1f}, \n2: {:.1f}, \n3: {:.1f}'.format(sum(reward_list1),
                                                                            sum(reward_list2),
                                                                            sum(reward_list3),
                                                                            sum(reward_list4)))
