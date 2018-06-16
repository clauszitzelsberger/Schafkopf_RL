# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:51:54 2018

@author: claus
"""

#from rules import Rules
#from state_overall import state_overall
#from state_player import state_player
from playing_schafkopf import playing_schafkopf

p = playing_schafkopf()

p.populate_memory()

p.training()
