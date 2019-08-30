import numpy as np

class PartiallyObservableMatrixGame:
  ''' turn-based stage game is kinda weird, so I reorganized it into an 'RL env-like' API
      still not happy with it but whatever... '''
  def __init__(self):
    self.n_cards = 2
    self.n_actions = 3
    self.cards_0 = None
    self.cards_1 = None
    self.player_1_act = None

    self.payoff_values = np.asarray([
        [
            [[10, 0, 0], [4, 8, 4], [10, 0, 0]],
            [[0, 0, 10], [4, 8, 4], [0, 0, 10]],
        ],
        [
            [[0, 0, 10], [4, 8, 4], [0, 0, 0]],
            [[10, 0, 0], [4, 8, 4], [10, 0, 0]],
        ],
    ], dtype=np.float32)

  def reset(self, one_hot=True):
    ''' resets state and returns player card observations '''
    self.cards_0 = np.random.choice(self.n_cards)
    self.cards_1 = np.random.choice(self.n_cards)
    self.player_1_act = None
    if one_hot:
      return [np.eye(self.n_cards)[self.cards_0], np.eye(self.n_cards)[self.cards_1]]
    else:
      return [self.cards_0, self.cards_1]
  
  def step(self, action, one_hot=True):
    if self.player_1_act is None:
      self.player_1_act = action
      return [np.eye(self.n_cards)[self.cards_0], np.eye(self.n_cards)[self.cards_1]], None, False
    else:
      rew = self.payoff_values[self.cards_0, self.cards_1, self.player_1_act, action]
      return [np.eye(self.n_cards)[self.cards_0], np.eye(self.n_cards)[self.cards_1]], rew, False
