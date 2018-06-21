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
        mistake = False #if player selects wrong game
        scores = [0, 0, 0, 0]

        self.state_overall = {'game': game,
                              'game_player': game_player,
                              'first_player': first_player,
                              'trick_number': trick_number,
                              'dealer': dealer,
                              'course_of_game': course_of_game,
                              'davongelaufen': davongelaufen,
                              'mistake': mistake,
                              'scores': scores}

        self.rules = Rules()

    def highest_card_of_played_cards(self):

        first_player=self.state_overall['first_player']
        tick_nr=self.state_overall['trick_number']
        game=self.state_overall['game']
        game_name=self.rules.name_of_game(game)

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
            if game_name[1] != 'wenz':
                if game_name[1] == 'sauspiel':
                    trump_color=2
                else: #solo
                    trump_color=game[0]
                trump_colors = self.rules.get_specific_cards(trumps, [trump_color, None])
                trump_colors = [[color, number] for color, number in trump_colors if number not in [4,5]]
                if len(trump_colors)>0:
                    trump_colors_numbers = [number for color, number in trump_colors]
                    highest_trump_color = [trump_color, max(trump_colors_numbers)]
                    return cards_list.index(highest_trump_color)

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
            opponents.remove(team_mate)
        else: #solo
            game_players = [game_player]

        total_score = sum([score_list[i] for i in game_players])

        # Basic reward depending on game type
        reward = self.rules.reward_basic[game[1]+1]

        # game players loose Schneider schwarz
        if total_score == self.rules.winning_thresholds[0]:
            winner = opponents
            reward += self.rules.reward_schneider[2]

        # game players loose Schneider
        elif self.rules.winning_thresholds[0] \
        < total_score <= \
        self.rules.winning_thresholds[1]:
            winner = opponents
            reward += self.rules.reward_schneider[1]

        # game players loose
        elif self.rules.winning_thresholds[1] \
        < total_score <= \
        self.rules.winning_thresholds[2]:
            winner = opponents
            reward += self.rules.reward_schneider[0]

        # game players win
        elif self.rules.winning_thresholds[2] \
        < total_score <= \
        self.rules.winning_thresholds[3]:
            winner = game_players
            reward += self.rules.reward_schneider[0]

        # game players win Schneider
        elif self.rules.winning_thresholds[3] \
        < total_score <= \
        self.rules.winning_thresholds[4]:
            winner = game_players
            reward += self.rules.reward_schneider[1]

        # game players win Schneider schwarz
        else:
            winner = game_players
            reward += self.rules.reward_schneider[2]

        if winner == game_players:
            f_winner=(4-len(winner)) / len(winner)
            f_looser=1
        else:
            f_winner=1
            f_looser=len(winner) / (4-len(winner))

        return [reward*f_winner if i in winner else -reward*f_looser for i in all]
