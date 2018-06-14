# state player greift mit self auf state_overall zu, hat aber somit nicht die infos die von playing schafkopf auf
# state_overall geschrieben werden und bekommt somit nur die default werte
# überprüfen ob es möglich ist, dass state_player nicht auf state_overview zugreift und die Daten als Argument
# übergeben bekommt

from rules import Rules
import numpy as np

class state_player():
    def __init__(self):
        player_id = None
        dealed_cards = [[None, None],
                        [None, None],
                        [None, None],
                        [None, None],
                        [None, None],
                        [None, None],
                        [None, None],
                        [None, None]]
        remaining_cards = [[None, None],
                           [None, None],
                           [None, None],
                           [None, None],
                           [None, None],
                           [None, None],
                           [None, None],
                           [None, None]]

        self.state_player = {'player_id': player_id,
                             'dealed_cards': dealed_cards,
                             'remaining_cards': remaining_cards}
        self.rules = Rules()

    def get_possible_cards_to_play(self, state_overall):

        cards_remaining=self.state_player['remaining_cards']
        cards_dealed=self.state_player['dealed_cards']
        possible_cards_to_play=[[color, number] for color, number \
                                in cards_remaining \
                                if color is not None and number is not None]
        trick_nr=state_overall.state_overall['trick_number']
        first_player=state_overall.state_overall['first_player']
        lead_card=state_overall.state_overall['course_of_game'][trick_nr][first_player]
        davongelaufen=state_overall.state_overall['davongelaufen']
        game=state_overall.state_overall['game']
        game_name = self.rules.name_of_game(game)[1]

        if game_name == 'sauspiel':
            rufsau = [game[0], 7]

            # No lead card available: one is first player
            if lead_card == [None, None]:

                # One doesn't have rufsau: one can lead every card
                if rufsau not in cards_remaining:
                    return possible_cards_to_play

                # One has rufsau
                else:
                    # One has more than 3 cards with the same color as rufsau's color: davonlaufen possible
                    # trumps aren't taken into account as they are colorless
                    """
                    if len([card for card in \
                            self.rules.get_specific_cards(cards_dealed, card=[game[0], None]) \
                            if card not in self.rules.get_trumps(game, cards_dealed)]) >= 4:
                    """
                    if len(self.rules.\
                        get_specific_cards2(cards_dealed, game, card=[game[0], None])) >=4:
                        return possible_cards_to_play

                    # One has less than 4 cards with the same color as rufsau's color: one can play rufsau and every
                    # other card excepts cards with rufsau's color
                    else:
                        possible_cards_to_play = \
                                [card for card in cards_remaining \
                                if card not in \
                                self.rules.get_specific_cards(cards_remaining, \
                                    card=[game[0], None])]
                        possible_cards_to_play.append(rufsau)
                        return possible_cards_to_play


            # There is already a lead card: one is 2nd - 4th player
            else:

                # Lead card is a trump
                if self.rules.get_trumps(game, [lead_card])==[lead_card]:

                    # One has at least one trump remaining: one has to play a trump
                    if len(self.rules.get_trumps(game, cards_remaining))>0:
                        possible_cards_to_play = self.rules.get_trumps(game, cards_remaining)
                        return possible_cards_to_play

                    # One doesn't have any trumps remaining
                    else:
                        # If one is davongelaufen: one can play every card remaining
                        if davongelaufen:
                            return possible_cards_to_play
                        # If rufsau still remaining: one can't play rufsau
                        else:
                            if trick_nr < 7 and rufsau in possible_cards_to_play:
                                possible_cards_to_play.remove(rufsau)
                                return possible_cards_to_play
                            else:
                                return possible_cards_to_play

                # Lead card has rufsau's color
                elif lead_card[0] == game[0] and \
                    self.rules.get_specific_cards(cards_remaining, card=rufsau)==rufsau:
                    if davongelaufen:
                        possible_cards_to_play=self.rules.get_specific_cards(cards_remaining, \
                            card=[lead_card[0], None])
                        return possible_cards_to_play

                    else:
                        possible_cards_to_play = [rufsau]
                        return possible_cards_to_play

                # Other cases
                else:

                    # One has lead card's color remaining
                    if len(self.rules.get_specific_cards2(cards_remaining, \
                    game, card=[lead_card[0], None])) > 0:
                        possible_cards_to_play=\
                            self.rules.get_specific_cards2(cards_remaining, \
                            game, [lead_card[0], None])
                        return possible_cards_to_play

                    # One doesn't have lead card's color
                    else:
                        # Same rule as above to be applied here
                        if davongelaufen:
                            return possible_cards_to_play
                        else:
                            if trick_nr < 7 and rufsau in possible_cards_to_play:
                                possible_cards_to_play.remove(rufsau)
                                return possible_cards_to_play
                            else:
                                return possible_cards_to_play

    def get_possible_games_to_play(self, state_overall):

        game_player = state_overall.state_overall['game_player']
        dealed_cards = self.state_player['dealed_cards']
        possible_games = self.rules.games[:3]
        possible_games.append(None)

        # No player defined: one is free to choose a game
        if game_player == None:
            # iterate over every color (except herz)
            for color in [0,1,3]:
                """
                cards_in_color=self.rules.get_specific_cards(dealed_cards, [color, None])
                trumps_in_color=\
                    self.rules.get_specific_cards(\
                    (self.rules.get_trumps([color, 0], dealed_cards)), [color, None])
                if (len(cards_in_color)-len(trumps_in_color))==0\
                    or [color, 7] in dealed_cards:
                    possible_games.remove([color, 0])
                """
                if len(self.\
                    rules.\
                    get_specific_cards2(cards_list=dealed_cards,
                    card=[color, None],
                    game=[color, 0]))==0\
                    or [color,7] in dealed_cards:
                    possible_games.remove([color, 0])
        else:
            possible_games = [None]
        return possible_games
