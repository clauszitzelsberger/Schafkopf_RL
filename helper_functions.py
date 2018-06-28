import matplotlib.pyplot as plt

def plot_reward(reward_list, reward_list2, reward_list3, reward_list4):
        x = range(len(reward_list))
        y = reward_list
        y2 = reward_list2
        y3 = reward_list3
        y4 = reward_list4
        plt.figure(figsize=(20,6))
        fig, ax = plt.subplots()
        ax.plot(x, y, 'black', alpha=.5, label='player 0 (RL bot)')
        ax.plot(x, y2, 'red', alpha=.5, label='player 1 (random)')
        ax.plot(x, y3, 'yellow', alpha=.5, label='player 2 (random)')
        ax.plot(x, y4, 'orange', alpha=.5, label='player 3 (random)')
        ax.set(xlabel='epochs x100', ylabel='avg reward',
               title='Reward ~ epochs')
        ax.legend()
        plt.show()

def plot_loss(loss_list):
    x = range(len(loss_list))
    y = loss_list
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel='epochs x100', ylabel='avg loss', title='Loss ~ epochs')
    plt.show()

