# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:51:54 2018

@author: claus
"""

#from rules import Rules
#from state_overall import state_overall
#from state_player import state_player
from train_select_game import train_select_game
from train_select_cards import train_select_cards


#t = train_select_game()
t = train_select_cards()

t.populate_memory()

t.training()
