import matplotlib.pyplot as plt
from rules import Rules

r = Rules()

def plot_reward(reward_list, reward_list2, 
                reward_list3, reward_list4,
                show_every, bool_show):
    if bool_show:
            x = range(len(reward_list))
            y = reward_list
            y2 = reward_list2
            y3 = reward_list3
            y4 = reward_list4
            plt.figure(figsize=(20,6))
            fig, ax = plt.subplots()
            ax.plot(x, y, 'black', alpha=.5, linewidth=5, label='player 0 (RL bot)')
            ax.plot(x, y2, 'red', alpha=.5, label='player 1 (random)')
            ax.plot(x, y3, 'yellow', alpha=.5, label='player 2 (random)')
            ax.plot(x, y4, 'orange', alpha=.5, label='player 3 (random)')
            ax.set(xlabel='epochs x{}'.format(show_every), ylabel='avg reward',
                   title='Reward ~ epochs')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()

def plot_loss(loss_list, loss_type, bool_show):
    if bool_show:
        x = range(len(loss_list))
        y = loss_list
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel='epochs x100', ylabel='avg loss', title='{} Loss ~ epochs'.format(loss_type))
        plt.show()

def return_course_of_game(state_overall, bool_show):
    if bool_show:
        game = r.name_of_game(state_overall['game'])
        game_player = state_overall['game_player']
        first_player = state_overall['first_player']
        trick_number = state_overall['trick_number']
        course_of_game = state_overall['course_of_game']
        cards = course_of_game[trick_number]
        #card0 = None
        #card1=None
        #card2=None
        #card3=None
        card_names=[None, None, None, None]
        for i in range(len(card_names)):
            if cards[i]!=[None, None]:
                card_names[i]=r.name_of_cards([cards[i]])[0]
            else:
                card_names[i]=[None, None]

        scores = state_overall['scores']

        fmt0 = '\t\t\t\t\t'
        fmt1 = fmt0 + 'Game: {} {}'
        fmt2 = fmt0 + 'Game Player: {}'
        fmt3 = fmt0 + 'Trick Nr: {}'
        fmt3b= fmt0 + 'Scores: {}'
        fmt4 = fmt0 + '{} {}'
        fmt5 = '{} {}' + fmt0 + fmt0 + '{} {}'
        print(fmt1.format(game[0], game[1]))
        print(fmt2.format(game_player))
        print(fmt3.format(trick_number))
        print(fmt3b.format(scores))
        print(fmt4.format(card_names[0][0], card_names[0][1]))
        print(fmt5.format(card_names[3][0], card_names[3][1], card_names[1][0], card_names[1][1]))
        print(fmt4.format(card_names[2][0], card_names[2][1]))
        print('\n\n')





