import numpy as np
from random import shuffle
import copy

class Rules():
    def __init__(self):
        self.card_number = ['siebener',
                            'achter',
                            'neuner',
                            'zehner',
                            'unter',
                            'ober',
                            'koenig',
                            'sau']

        self.card_color = ['eichel', 'gras', 'herz', 'schellen']

        self.card_scores = [0,0,0,10,2,3,4,11]

        ############## eichel # gras # herz # schellen #
        self.cards = [[0,0], [1,0], [2,0], [3,0], #siebener
                      [0,1], [1,1], [2,1], [3,1], #achter
                      [0,2], [1,2], [2,2], [3,2], #neuner
                      [0,3], [1,3], [2,3], [3,3], #zehner
                      [0,4], [1,4], [2,4], [3,4], #unter
                      [0,5], [1,5], [2,5], [3,5], #ober
                      [0,6], [1,6], [2,6], [3,6], #koenig
                      [0,7], [1,7], [2,7], [3,7]] #sau

        self.game_names = ['sauspiel', 'solo', 'wenz']

        ############# eichel # gras # herz # schellen #
        self.games = [[None, None],               #no game
                      [0,0], [1,0],        [3,0], #sauspiel
                      [0,1], [1,1], [2,1], [3,1], #solo
                      [None, 2]]                  #wenz

        self.reward_basic = [0, 20, 50, 50] # no game, sauspiel, solo, wenz
        self.reward_schneider = [0, 10, 20] # normal, schneider, schneider schwarz

        self.winning_thresholds = [0, 30, 60, 90, 119]

    def shuffle_cards(self):
        cards = copy.copy(self.cards)
        shuffle(cards)
        return cards

    def deal_cards(self, number_of_players=4):
        shuffled_cards = self.shuffle_cards()
        return [shuffled_cards[:8],
                shuffled_cards[8:16],
                shuffled_cards[16:24],
                shuffled_cards[24:32]]

    def name_of_cards(self, cards_list):
        return [[self.card_color[color], self.card_number[number]] \
                for color, number in cards_list]

    def name_of_game(self, game):
        color = game[0]
        game_type = game[1]
        
        if color == None:
            return [None, self.game_names[game_type]]
        else:
            return [self.card_color[color], self.game_names[game_type]]

    def get_specific_cards(self, cards_list, card=[None, None]):
        if card[0] == None and card[1] == None:
            return cards_list
        if card[0] != None and card[1] != None:
            if card in cards_list:
                return card
            else:
                return []
        if card[0] != None:
            return [[color, number] for color, number in cards_list if (color in [card[0]])]
        if card[1] != None:
            return [[color, number] for color, number in cards_list if (number in [card[1]])]

    def get_trumps(self, game, cards_list):
        if self.name_of_game(game)[1] == 'sauspiel':
            trump_colors = [2] #Herz
            trump_numbers = [4, 5] # Unter, Ober

        elif self.name_of_game(game)[1] == 'solo':
            trump_colors = [game[0]]
            trump_numbers = [4, 5]
        else: #wenz
            trump_colors = []
            trump_numbers = [4]
        return [[color, number] for color, number in cards_list \
            if color in trump_colors or number in trump_numbers]

    def get_specific_cards2(self,
                            cards_list,
                            game,
                            card=[None, None],
                            wo_trumps=True):
        if wo_trumps:
            cards_in_color = self.get_specific_cards(cards_list, [card[0], None])
            trumps = self.get_trumps(game, cards_list)
            return [cards for cards in cards_in_color \
                if cards not in trumps]
        else:
            return self.get_specific_cards(cards_list, card)

    def get_index(self, item, type):
        """
        Get index either cards or games list
        Params:
            item: card or game
            type: string if item is card or game
        """
        if type == 'card':
            return self.cards.index(item)
        else:
            return self.games.index(item)

    def get_one_hot_cards(self, cards_list_indexed):
        number_cards = len(self.cards)
        return_ =  [int(i in cards_list_indexed) for i in range(number_cards)]
        return return_

    def get_one_hot_games(self, game_indexed):
        number_games = len(self.games)
        return [int(i in game_indexed) for i in range(number_games)]

    def get_one_hot(self, subgroup, len_group):
        return [int(i in subgroup) for i in range(len_group)]
