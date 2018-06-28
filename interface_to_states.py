from rules import Rules
from state_overall import state_overall
from state_player import state_player
import numpy as np
#import random
import copy


class interface_to_states():
    def __init__(self):
        self.rules = Rules()

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
        for i in range(len(self.players)):
            self.players[i].state_player['dealed_cards']=dealed_cards[i]
            self.players[i].state_player['remaining_cards']=dealed_cards[i]

    def order_players(self):
        first_player = self.state_overall.state_overall['first_player']

        # order players
        players = [self.players[first_player],
                   self.players[(first_player+1)%4],
                   self.players[(first_player+2)%4],
                   self.players[(first_player+3)%4]]
        return players

    def return_possible_games(self, player_i):
        players = self.order_players()
        return players[player_i].get_possible_games_to_play(self.state_overall)

    def write_game_to_states(self, selected_game, player_i):
        first_player = self.state_overall.state_overall['first_player']
        self.state_overall.state_overall['game']=selected_game
        self.state_overall.state_overall['game_player']=(first_player+player_i)%4

    def return_possbile_cards(self, player_i):
        players = self.order_players()
        return players[player_i].get_possible_cards_to_play(self.state_overall)

    def write_card_to_states(self, selected_card, player_i, davongelaufen=False):
        players = self.order_players()
        trick_nr = self.state_overall.state_overall['trick_number']
        first_player = self.state_overall.state_overall['first_player']
        remaining_cards=copy.copy(players[player_i].state_player['remaining_cards'])
        remaining_cards.remove(selected_card)
        players[player_i].state_player['remaining_cards'] = remaining_cards
        self.state_overall.state_overall['course_of_game'][trick_nr][(first_player+player_i)%4] = \
            selected_card
        # Not implemented yet: logic missing if davongelaufen True or False
        if davongelaufen:
            self.state_overall.state_overall['davongelaufen'] = davongelaufen
    
    def update_first_player_trick_nr_score(self):
        highest_card = self.state_overall.highest_card_of_played_cards()
        self.state_overall.state_overall['first_player'] = \
            highest_card
        self.state_overall.state_overall['scores'][highest_card] += \
            self.state_overall.get_score()
        self.state_overall.state_overall['trick_number'] += 1

    def return_state_overall(self):
        return self.state_overall.state_overall

    def return_state_player(self, player_id):
        return self.players[player_id].state_player

    def return_reward_list(self):
        return self.state_overall.get_reward_list()

    def return_state_select_card(self, player_id):
        """
        returns complete state space for a player given his id
            state_size: 1086
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
        """
        state_overall = self.return_state_overall()
        state_player = self.return_state_player(player_id)
        state_list =[]

        game = state_overall['game']
        state_list.extend(self.rules.get_one_hot_games([self.rules.get_index(game, 'game')]))

        game_player = state_overall['game_player']
        state_list.extend(self.rules.get_one_hot([game_player], 4))

        first_player = state_overall['first_player']
        state_list.extend(self.rules.get_one_hot([first_player], 4))

        trick_nr = state_overall['trick_number']
        state_list.extend(self.rules.get_one_hot([trick_nr], 8))

        course_of_game = state_overall['course_of_game']
        course_of_game = np.array(course_of_game)
        course_of_game = np.squeeze(course_of_game.reshape(8*4,-1,2), axis=1)
        course_of_game = course_of_game.tolist()
        for card in course_of_game:
            if card == [None, None]:
                state_list.extend([0]*32)
            else:
                state_list.extend(self.rules.get_one_hot_cards([self.rules.get_index(card, 'card')]))

        state_list.extend([int(state_overall['davongelaufen'])])

        state_list.extend(state_overall['scores'])

        remaining_cards = state_player['remaining_cards']
        remaining_cards_indexed = [self.rules.get_index(card, 'card') for card in remaining_cards]
        state_list.extend(self.rules.get_one_hot_cards(remaining_cards_indexed))

        return state_list


