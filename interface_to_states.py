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
        return [self.state_overall.get_reward_list(), self.state_overall.get_team_mate()]

    def return_state_select_card(self, player_id):
        """
        returns complete state space for a player given his id
            state_size: 1086 | 225
            - state_overall:
                - player_id:4
                - game: 9
                - game_player: 4
                - first_player: 4
                - trick_nr: 8
                - course_of_game: 32*32=1024 or other options + current trick
                - davongelaufen: 1
                - scores: 4
            - state_player:
                - remaining_cards: 32
        """
        state_overall = self.return_state_overall()
        state_player = self.return_state_player(player_id)
        state_list =[]

        #state_list.extend(self.rules.get_one_hot([player_id],4))

        game = state_overall['game']
        state_list.extend(self.rules.get_one_hot_games([self.rules.get_index(game, 'game')]))

        # Game player (relative to observed player)
        game_player = state_overall['game_player']
        state_list.extend(self.rules.get_one_hot([(game_player-player_id)%4], 4))

        first_player = state_overall['first_player']
        state_list.extend(self.rules.get_one_hot([(first_player-player_id)%4], 4))

        trick_nr = state_overall['trick_number']
        #state_list.extend(self.rules.get_one_hot([trick_nr], 8))

        course_of_game = state_overall['course_of_game']
        course_of_game = np.array(course_of_game)


        # Reorder course of game to get relative order for observed player
#        course_of_game = np.swapaxes(course_of_game, 0, 1)
#        course_of_game = np.array([course_of_game[(player_id+i)%4] for i in np.arange(0,4)])
#        course_of_game = np.swapaxes(course_of_game, 1, 0)
#
#        course_of_game = np.squeeze(course_of_game.reshape(8*4,-1,2), axis=1)
#        course_of_game = course_of_game.tolist()
        
        # Option 1: exact course of game as state
#        for card in course_of_game:
#            if card == [None, None]:
#                state_list.extend([0]*32)
#            else:
#                state_list.extend(self.rules.get_one_hot_cards([self.rules.get_index(card, 'card')]))
        
        #state_list.extend([int(state_overall['davongelaufen'])])

        # Option 2: cards already played
#        played_cards = [card for card in course_of_game if card != [None, None]]
#        played_cards_indexed = [self.rules.get_index(card, 'card') for card in played_cards]
#        state_list.extend(self.rules.get_one_hot_cards(played_cards_indexed))
#        # Cards played in this trick
#        cards_in_trick = state_overall['course_of_game'][trick_nr]
#        cards_in_trick = [card for card in cards_in_trick if card != [None, None]]
#        cards_in_trick_indexed = [self.rules.get_index(card, 'card') for card in cards_in_trick]
#        state_list.extend(self.rules.get_one_hot_cards(cards_in_trick_indexed))


        # Option 3: cards already played by each player
        # First order cards relatively to observed player
        course_of_game = np.swapaxes(course_of_game, 0, 1)
        course_of_game = np.array([course_of_game[(player_id+i)%4] for i in np.arange(0,4)])
        course_of_game = course_of_game.tolist()
                                          
        for p in np.arange(0,4):
            # Alocate cards to players
            played_cards = [card for card in course_of_game[p] if card != [None, None]]
            played_cards_indexed = [self.rules.get_index(card, 'card') for card in played_cards]
            state_list.extend(self.rules.get_one_hot_cards(played_cards_indexed))
        
        # Cards played in this trick
        cards_in_trick = state_overall['course_of_game'][trick_nr]
        # order cards
        cards_in_trick = [cards_in_trick[(player_id+i)%4] for i in np.arange(0,4)]
        
        for card in cards_in_trick:
            if card == [None, None]:
                state_list.extend([0]*32)
            else:
                state_list.extend(self.rules.get_one_hot_cards([self.rules.get_index(card, 'card')]))

        #state_list.extend([s/120 for s in state_overall['scores']])

        #remaining_cards = state_player['remaining_cards']
        #remaining_cards_indexed = [self.rules.get_index(card, 'card') for card in remaining_cards]
        #state_list.extend(self.rules.get_one_hot_cards(remaining_cards_indexed))

        # Make use of state for select game
        state_list.extend(self.return_state_select_game(player_id, state_player['remaining_cards']))

        

        return state_list

    def return_state_select_game(self, player_id, cards_list=None):
        """
        return complete state for a player given his id:
        States space to be a vector:
        - Number of cards for every color within [7,9]: 4
        - -"- [Koenig, 10]: 4
        - Boolean if Sau is available for every color: 4
        - Number of Unter: 4
        - Nummber of Ober: 4
        """
        if cards_list==None:
            dealed_cards = self.return_state_player(player_id)['dealed_cards']
        else:
            dealed_cards = cards_list

        state_list=[]

        colors=np.arange(0,4)

        # [7,9]
        for color in colors:
            number_cards=0
            for number in np.arange(0,3):
                number_cards += len(self.rules.get_specific_cards(dealed_cards, [color, number]))
            number_cards = number_cards/3.0
            state_list.append(number_cards)

        # [Koenig, 10]
        for color in colors:
            nummber_cards=0
            for number in [3,6]:
                number_cards += len(self.rules.get_specific_cards(dealed_cards, [color, number]))
            number_cards = number_cards/2.0
            state_list.append(number_cards)

        # Sau
        for color in colors:
            state_list.append(len(self.rules.get_specific_cards(dealed_cards, [color, 7]))/4.0)

        # Unter
        for color in colors:
            state_list.append(len(self.rules.get_specific_cards(dealed_cards, [color, 4]))/4.0)

        # Ober
        for color in colors:
            state_list.append(len(self.rules.get_specific_cards(dealed_cards, [color, 5]))/4.0)
            
        return state_list




