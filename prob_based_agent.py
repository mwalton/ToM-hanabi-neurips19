import pyhanabi
from rl_env import Agent
from random import randint
import numpy as np
import copy


class ProbAgent(Agent):
  """Agent that applies a simple heuristic."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)

  @staticmethod
  def playable_card(card, fireworks):
    """A card is playable if it can be placed on the fireworks pile."""
    return card['rank'] == fireworks[card['color']]

  def get_new_player_observed_hands(self, observation, c_obs, replaced_player_id, replacement_hand, their_id):
    hand_list = []
    c_card = pyhanabi.ffi.new("pyhanabi_card_t*")
    for pid in range(c_obs.num_players()):
      player_hand = []
      hand_size = pyhanabi.lib.ObsGetHandSize(c_obs._observation, pid)
      if pid == replaced_player_id:
        for card in replacement_hand:
          player_hand.append({'color': pyhanabi.color_idx_to_char(int(card / 5)), 'rank': card % 5})
      elif pid == their_id:
        for card in replacement_hand:
          player_hand.append({'color': None, 'rank': -1})
      else:
        for i in range(hand_size):
          pyhanabi.lib.ObsGetHandCard(c_obs._observation, pid, i, c_card)
          player_hand.append(pyhanabi.HanabiCard(c_card.color, c_card.rank).to_dict())
      
      hand_list.append(player_hand)
    return hand_list

  def level_one_belief(self, num_samples, probs, self_card_knowledge, self_id, other_card_knowledges, other_id, my_obs, their_obs, other_c_obs):
    discards = my_obs['discard_pile']
    # their_observed_hands = copy.deepcopy(their_obs['observed_hands'])
    fireworks = my_obs['fireworks']
    prob_vecs = np.zeros(shape=(len(self_card_knowledge), 25), dtype=float)
    attempts = 0
    while attempts < 100:
      ind = 0
      while ind < num_samples:
        sample_hand = []
        for prob in probs:
          card_prob = np.array(prob, dtype=float) / np.sum(prob, dtype=float)
          sample_hand.append(np.random.choice(25, 1, p=card_prob)[0])
        
        hand_prob = self.hand_prob(probs, self_card_knowledge, sample_hand)

        # for card_ind, card in enumerate(sample_hand):
        #   their_observed_hands[self_id][card_ind]['rank'] = card % 5
        #   their_observed_hands[self_id][card_ind]['color'] = pyhanabi.color_idx_to_char(card // 5)

        their_observed_hands = self.get_new_player_observed_hands(their_obs, other_c_obs, self_id, sample_hand, other_id)

        new_probs = np.array(self.card_knowledge_to_prob_vectors(other_card_knowledges, discards, their_observed_hands, fireworks, as_counts=False))
        new_probs *= hand_prob

        prob_vecs += new_probs
        ind += 1
      attempts += 1
      if np.sum(prob_vecs) > 0.:
        break
      
    ret_list = []
    for prob in prob_vecs:
      prob /= np.sum(prob)
      ret_list.append(list(prob))
    return ret_list

  @staticmethod
  def hand_prob(prob_counts, self_knowledge, hand):
    knowledge_set = set(self_knowledge)
    match_inds = [[] for _ in knowledge_set]
    for ind, knowledge in enumerate(knowledge_set):
      for query_ind, query_knowledge in enumerate(self_knowledge):
        if query_knowledge == knowledge:
          match_inds[ind].append(query_ind)

    prob = 1.

    for ind_list in match_inds:
      for ind in ind_list:
        if np.sum(prob_counts[ind_list[0]], dtype=float) == 0.:
          return 0.
        card_prob = np.array(prob_counts[ind_list[0]], dtype=float) / np.sum(prob_counts[ind_list[0]], dtype=float)
        card_ind = np.random.choice(25, 1, p=card_prob)[0]
        prob_counts[ind_list[0]][card_ind] = prob_counts[ind_list[0]][card_ind] - 1
        if prob_counts[ind_list[0]][card_ind] < 0:
          return 0.
        prob *= card_prob[card_ind]

    return prob

  @staticmethod
  def card_knowledge_to_prob_vectors(card_knowledges, discards, observed_hands, fireworks, as_counts=False):
    # discards = observation['discard_pile']
    # observed_hands = observation['observed_hands']
    # fireworks = observation['fireworks']
    infosets = []
    for card_knowledge in card_knowledges:
      colors = []
      ranks = []
      valid_info = card_knowledge.split('|')[1]
      for card_info in valid_info:
        if card_info in pyhanabi.COLOR_CHAR:
          colors.append(pyhanabi.color_char_to_idx(card_info))
        else:
          ranks.append(int(card_info) - 1)

      # Store indices for length 50 vectors that will hold card counts/probs that should
      # be updated using card counts
      infoset = []
      for color in colors:
        for rank in ranks:
          infoset.append((5 * color) + rank)

      infosets.append(infoset)
    
    card_counts = [0]*25
    for ind in range(len(card_counts)):
      if ind % 5 == 0:
        card_counts[ind] = 3
      elif ind % 5 < 4:
        card_counts[ind] = 2
      else:
        card_counts[ind] = 1

    for card in discards:
      card_counts[(5 * pyhanabi.color_char_to_idx(card['color'])) + card['rank']] -= 1

    for color, rank in fireworks.items():
      if rank > 0:
        for ind in range(rank):
          card_counts[(5 * pyhanabi.color_char_to_idx(color)) + ind] -= 1

    for hand in observed_hands:
      for card_info in hand:
        if card_info['rank'] < 0:
          break
        card_counts[(5 * pyhanabi.color_char_to_idx(card_info['color'])) + card_info['rank']] -= 1

    prob_vecs = [[0.]*25 for _ in range(len(infosets))]
    for set_ind, infoset in enumerate(infosets):
      set_sum = 0.
      for ind in infoset:
        set_sum += float(card_counts[ind])
        if card_counts[ind] > 0:
          prob_vecs[set_ind][ind] = float(card_counts[ind])
    
      if not as_counts and set_sum > 0:
        for ind in range(25):
          prob_vecs[set_ind][ind] /= set_sum
    
    return prob_vecs

  def act(self, observation):
    """Act based on an observation."""
    if observation['current_player_offset'] != 0:
      return None

    # Check if there are any pending hints and play the card corresponding to
    # the hint.
    for card_index, hint in enumerate(observation['card_knowledge'][0]):
      if hint['color'] is not None or hint['rank'] is not None:
        return {'action_type': 'PLAY', 'card_index': card_index}

    # Check if it's possible to hint a card to your colleagues.
    fireworks = observation['fireworks']
    if observation['information_tokens'] > 0:
      # Check if there are any playable cards in the hands of the opponents.
      for player_offset in range(1, observation['num_players']):
        player_hand = observation['observed_hands'][player_offset]
        player_hints = observation['card_knowledge'][player_offset]
        # Check if the card in the hand of the opponent is playable.
        for card, hint in zip(player_hand, player_hints):
          # (ASF) Added for testing
          if ProbAgent.playable_card(card, fireworks):
            if hint['color'] is None and hint['rank'] is None:
              if randint(0, 1) > 0:
                return {
                    'action_type': 'REVEAL_COLOR',
                    'color': card['color'],
                    'target_offset': player_offset
                }
              else:
                return {
                    'action_type': 'REVEAL_RANK',
                    'rank': card['rank'],
                    'target_offset': player_offset
                }

    # If no card is hintable then discard or play.
    if observation['information_tokens'] < self.max_information_tokens:
      return {'action_type': 'DISCARD', 'card_index': 0}
    else:
      return {'action_type': 'PLAY', 'card_index': 0}
