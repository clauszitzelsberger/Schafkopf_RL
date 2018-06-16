from rules import Rules
import numpy as np
import random

class state_overall():
    def __init__(self):
        dealer = random.choice([0,1,2,3])
        game = [None, None]
        game_player = None
        first_player = (dealer+1)%4
        trick_number = 0
        #################### player 1 ### player 2 ### player 3 ### player 4
        course_of_game = [[[None, None], [None, None], [None, None], [None, None]], # 1. trick
                          [[None, None], [None, None], [None, None], [None, None]], # 2. trick
                          [[None, None], [None, None], [None, None], [None, None]], # ...
                          [[None, None], [None, None], [None, None], [None, None]],
                          [[None, None], [None, None], [None, None], [None, None]],
                          [[None, None], [None, None], [None, None], [None, None]],
                          [[None, None], [None, None], [None, None], [None, None]],
                          [[None, None], [None, None], [None, None], [None, None]]]
        davongelaufen = False
        scores = [0, 0, 0, 0]

        self.state_overall = {'game': game,
                              'game_player': game_player,
                              'first_player': first_player,
                              'trick_number': trick_number,
                              'dealer': dealer,
                              'course_of_game': course_of_game,
                              'davongelaufen': davongelaufen,
                              'scores': scores}

        self.rules = Rules()

    def highest_card_of_played_cards(self):

        first_player=self.state_overall['first_player']
        tick_nr=self.state_overall['trick_number']
        game = self.state_overall['game']
        game_name = self.rules.name_of_game(game)[1]

        cards_list=self.state_overall['course_of_game'][tick_nr]
        lead_card=cards_list[first_player]
        lead_card_color=lead_card[0]
        trumps=self.rules.get_trumps(game, cards_list)


        if len(trumps) > 0:
            ober = self.rules.get_specific_cards(trumps, [None, 5])
            if len(ober)>0:
                ober_colors = [color for color, number in ober]
                highest_ober = [min(ober_colors), 5]
                return cards_list.index(highest_ober)
            unter = self.rules.get_specific_cards(trumps, [None, 4])
            if len(unter)>0:
                unter_colors = [color for color, number in unter]
                highest_unter = [min(unter_colors), 4]
                return cards_list.index(highest_unter)
            herz = self.rules.get_specific_cards(trumps, [2, None])
            herz = [[color, number] for color, number in herz if number not in [4,5]]
            if len(herz)>0:
                herz_numbers = [number for color, number in herz]
                highest_herz = [2, max(herz_numbers)]
                return cards_list.index(highest_herz)

        # No trumps in cards
        else:
            cards_in_lead_card_color=\
                self.rules.get_specific_cards(cards_list, [lead_card_color, None])
            numbers = [number for color, number in cards_in_lead_card_color]
            highest_played_color = [lead_card_color, max(numbers)]
            return cards_list.index(highest_played_color)

    def get_team_mate(self):

        game = self.state_overall['game']
        game_name = self.rules.name_of_game(game)[1]

        if game_name == 'sauspiel':
            rufsau = [game[0], 7]
            course_of_game = self.state_overall['course_of_game']
            course_of_game = np.array(course_of_game)
            course_of_game = np.squeeze(course_of_game.reshape(8*4,-1,2), axis=1)
            return course_of_game.tolist().index(rufsau)%4
        else:
            return None

    def get_score(self):
        trick_nr = self.state_overall['trick_number']
        cards_in_trick = self.state_overall['course_of_game'][trick_nr]
        return sum([self.rules.card_scores[number] for color, number in cards_in_trick])

    def get_reward_list(self):
        game = self.state_overall['game']
        game_name = self.rules.name_of_game(game)[1]
        game_player = self.state_overall['game_player']
        all = [0,1,2,3]
        opponents = [0,1,2,3]
        opponents.remove(game_player)
        score_list = self.state_overall['scores']

        if game_name == 'sauspiel':
            team_mate = self.get_team_mate()
            game_players = [game_player, team_mate]
            total_score = sum([score_list[i] for i in game_players])

            # game players loose Schneider schwarz
            if total_score == 0:
                winner = opponents
                reward = 30

            # game players loose Schneider
            elif 0 < total_score <= 30:
                winner = opponents
                reward = 20

            # game players loose
            elif 30 < total_score <= 60:
                winner = opponents
                reward = 10

            # game players win
            elif 60 < total_score <= 90:
                winner = game_players
                reward = 10

            # game players win Schneider
            elif 90 < total_score <=119:
                winner = game_players
                reward = 20

            # game players win Schneider schwarz
            else:
                winner = game_players
                reward = 30

        return [reward if i in winner else -reward for i in all]
