# Schafkopf_RL
Reinforcement Learning applied on the Bavarian card game 'Schafkopf'

## Set up:
| module               | content                                                          | 
| -------------------- |:----------------------------------------------------------------:|
| rules.py             | definition of cards, scores, games, rewards and helper methods   |
| state_overall.py     | states which are valid for every player                          |
| state_player.py      | states which are valid for single player                         |
| playing_schaflopf.py | hyperparameters for NNs, training Q Networks                     |
| QL_choose_game.py    | Q Network achitechture and Memory for choosing game              |
| QL_select_card.py    | *not yet implemented*                                            |
| script.py            | script                                                           |

## Next steps:
| Action                                     |                                                |
| -------------------------------------------|:-----------------------------------------------|
| set state davongelaufen                    | identify if player davongelaufen in play()     |
| QL for selecting cards                     |                                                |
| QL for doppeln                             |                                                |
| Stock                                      |                                                |
| select game process right now simplified   |                                                |
| Tout, Sie                                  |                                                |

## Results:
### Reward ~ epochs after implementing QL for choosing game
Player 1 = RL bot   
Player 2-4 = acting random   
![alt text][logo]

[logo]: https://github.com/clauszitzelsberger/Schafkopf_RL/tree/master/plots/reward_epochs_select_game.PNG "Reward~Epochs"
