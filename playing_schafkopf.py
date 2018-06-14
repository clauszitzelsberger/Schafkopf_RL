from rules import Rules
from state_overall import state_overall
from state_player import state_player
from QL_choose_game import QNetwork
from QL_choose_game import Memory
import numpy as np
import random
import copy
import tensorflow as tf

class playing_schafkopf():
    def __init__(self):
        self.rules = Rules()
        self.state_overall = state_overall()



        self.QNetwork = QNetwork()
        #self.Memory = Memory()

        self.train_episodes = 1000          # max number of episodes to learn from
        self.max_steps = 200                # max steps in an episode
        self.gamma = 0.99                   # future reward discount

        # Exploration parameters
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.01            # minimum exploration probability
        self.decay_rate = 0.0001            # exponential decay rate for exploration prob

        # Network parameters
        self.hidden_size = 64               # number of units in each Q-network hidden layer
        self.learning_rate = 0.0001         # Q-network learning rate

        # Memory parameters
        self.memory_size = 10000            # memory capacity
        self.batch_size = 20                # experience mini-batch size
        self.pretrain_length = self.batch_size   # number experiences to pretrain the memory

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

    def dealing(self):
        dealed_cards = self.rules.deal_cards()
        print(dealed_cards)
        for i in range(len(self.players)):
            self.players[i].state_player['dealed_cards']=dealed_cards[i]
            self.players[i].state_player['remaining_cards']=dealed_cards[i]

    def new_game_to_choose(self):

        first_player = self.state_overall.state_overall['first_player']

        # order players
        players = [self.players[first_player],
                   self.players[(first_player+1)%4],
                   self.players[(first_player+2)%4],
                   self.players[(first_player+3)%4]]

        for i in range(len(players)):
            possible_games = players[i].\
                get_possible_games_to_play(self.state_overall)
            selected_game = random.choice(possible_games)
            if selected_game != None:
                self.state_overall.state_overall['game']=selected_game
                self.state_overall.state_overall['game_player']=i
                break

    def play(self):

        trick_nr = self.state_overall.state_overall['trick_number']

        # oder players
        first_player = self.state_overall.state_overall['first_player']

        players = [self.players[first_player],
                   self.players[(first_player+1)%4],
                   self.players[(first_player+2)%4],
                   self.players[(first_player+3)%4]]

        for i in range(len(players)):
            possible_cards = players[i].\
                get_possible_cards_to_play(self.state_overall)
            selected_card = random.choice(possible_cards)
            print('Possible Cards: {}'.format(possible_cards))
            print('Selected Cards: {}'.format(selected_card))

            # Update states for next player
            remaining_cards=copy.copy(players[i].state_player['remaining_cards'])
            remaining_cards.remove(selected_card)
            remaining_cards.append([None, None])
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

    def populate_memory(self.):
        memory = Memory(max_size=self.memory_size)

        # Make random actions and store experiences
        for i in range(self.pretrain_length):
            self.reset_epsiode()
            self.dealing()
            new_game_to_choose()

            # Action
            game = self.state_overall.state_overall['game']
            action = self.rule.get_index(game, 'game')

            # State
            game_player = self.state_overall.state_overall['game_player']
            state = self.players[game_player]['dealed_cards']
            state = [self.rule.get_index(card, 'card') for card in state]

            if self.state_overall.state_overall['game'] != [None, None]:
                while self.state_overall.state_overall['trick_number'] < 8:
                    self.play()
                rewards = self.state_overall.get_reward_list()
            else:
                rewards = [0,0,0,0]

            # Reward
            reward = rewards[game_player]
            memory.add(state, action, reward)


    def training(self):
        self.reset_epsiode()
        self.dealing()
        self.new_game_to_choose()
        if self.state_overall.state_overall['game'] != [None, None]:
            while self.state_overall.state_overall['trick_number'] < 8:
                self.play()
            rewards = self.state_overall.get_reward_list()
            print(rewards)
            return rewards
        return [0,0,0,0] #no game, reward is 0 for every player
        #__init__()
