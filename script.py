# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:51:54 2018

@author: claus
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#from rules import Rules
#from state_overall import state_overall
#from state_player import state_player
#from train_select_game import train_select_game
#from train_select_cards import train_select_cards
#from train_select_both import train_select_both
from train_select_both_apply import train_select_both_apply

#t = train_select_game()
#t = train_select_cards()
#t = train_select_both()
t = train_select_both_apply()

#t.populate_memory()

#t.training()
#t.apply_model()

t.populate_memoryC()
t.populate_memoryG()
t.training()
#t.apply_model()
