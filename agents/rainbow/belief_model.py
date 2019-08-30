import pyhanabi
import numpy as np
from copy import copy
from scipy.stats import entropy, wasserstein_distance

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('{}: {}'.format(f.__name__, end-start))
        return result
    return wrapper

KL_FIX_SCALAR = 0.000000001
DEFAULT_CARD_COUNTS = [0] * 25

for ind in range(25):
  if ind % 5 == 0:
    DEFAULT_CARD_COUNTS[ind] = 3
  elif ind % 5 < 4:
    DEFAULT_CARD_COUNTS[ind] = 2
  else:
    DEFAULT_CARD_COUNTS[ind] = 1

class HandBeliefModel(object):
  def __init__(self, belief_level, modification_method, n_players, num_b0_samples=1, comms_reward=False, beta=0.0):
    self.belief_level = belief_level
    self.modification_method = modification_method
    self.n_cards = 5 if n_players in (1, 2, 3) else 4
    self.n_players = n_players
    self.num_b0_samples = num_b0_samples
    self.comms_reward = comms_reward
    self.beta = beta

  def _entropy_map(self, p, q):
    kl = sum([entropy(p1, q1 + KL_FIX_SCALAR) for p1, q1 in zip(p, q) if np.sum(p1) > 0])
    if np.isnan(kl) or np.isinf(kl):
      print("NAN")

    kl = max(0., min(kl, 25.))
    return kl

  def _wasserstein_map(self, p, q):
    kl = sum([wasserstein_distance(p1, q1) for p1, q1 in zip(p, q) if np.sum(p1) > 0])
    return kl

  def get_reward_for_hand(self, current_cards, current_belief, next_cards, next_belief):
    # div_score = self._entropy_map(current_cards, current_belief) - self._entropy_map(next_cards, next_belief)
    div_score = self._wasserstein_map(current_cards, current_belief) - self._wasserstein_map(next_cards, next_belief)
    return div_score

  def get_comms_reward(self, observation_vector, next_observation_vector):
    if not self.comms_reward:
      return 0.0
    else:
      current_belief, current_player_hands = self.obs_to_hands(observation_vector)
      next_belief, next_player_hands = self.obs_to_hands(next_observation_vector)
      div_reward = 0.
      for current_cards, current_belief, next_cards, next_belief in zip(current_player_hands, current_belief, next_player_hands, next_belief):
        div_reward += self.get_reward_for_hand(current_cards, current_belief, next_cards, next_belief)
      
      return div_reward * self.beta
  
  def obs_to_hands(self, observation_vector):
    hand_vec_len = self.n_cards * 25 * (self.n_players - 1)
    player_hands_vec = observation_vector[:hand_vec_len]
    belief_vec = observation_vector[-hand_vec_len:]
    belief = np.reshape(belief_vec, (self.n_players - 1, self.n_cards, 25))
    player_hands = np.reshape(player_hands_vec, (self.n_players - 1, self.n_cards, 25))

    return belief, player_hands

  def belief_size(self):
    if self.belief_level == -1:
      return 0
    elif self.belief_level == 0:
      return 25 * self.n_cards
    elif self.belief_level == 1:
      return 25 * self.n_cards * self.n_players
    else:
        raise ValueError("Invalid belief level; nested belief only from lvl -1, 0, 1")

  def get_belief(self, observation, agent_id):
    if self.belief_level == -1:
        return None
    elif self.belief_level == 0:
        return self.belief_lvl0(observation, agent_id)
    elif self.belief_level == 1:
        return self.belief_lvl1(observation, agent_id)
    else:
        raise ValueError("Invalid belief level; nested belief only from lvl -1, 0, 1")

  def modify_observation(self, observations, observation_vector, agent_id):
      if self.belief_level == -1:
          return observation_vector

      if self.modification_method == 'concat' or self.modification_method == 'replace':
        if self.belief_level == 0:
          probas = self.get_belief(observations, agent_id)
          return np.concatenate((observation_vector, probas.flatten()))
        if self.belief_level == 1:
          lvl0_belief, lvl1_belief = self.belief_lvl1(observations, agent_id)
          return np.concatenate((observation_vector, lvl0_belief.flatten(), lvl1_belief.flatten()))
      else:
        raise ValueError("Invalid observation modification method")

  def belief_lvl0(self, observations, agent_id):
    c_obs = observations['player_observations'][agent_id]['pyhanabi']
    obs = observations['player_observations'][agent_id]
    c_knowledge = [str(x) for x in c_obs.card_knowledge()[0]]
    probs = self.card_knowledge_to_prob_vectors(c_knowledge, obs['discard_pile'], obs['observed_hands'], obs['fireworks'], False)
    return probs
  
  def belief_lvl1(self, observations, agent_id):
    c_obs = observations['player_observations'][agent_id]['pyhanabi']
    obs = observations['player_observations'][agent_id]
    c_knowledge = [str(x) for x in c_obs.card_knowledge()[0]]
    lvl0_counts = self.card_knowledge_to_prob_vectors(c_knowledge, obs['discard_pile'], obs['observed_hands'], obs['fireworks'], True)

    lvl1_probs = []
    for other_agent_id in range(len(observations['player_observations'])):
      if other_agent_id != agent_id:
        other_c_obs = observations['player_observations'][other_agent_id]['pyhanabi']
        other_obs = observations['player_observations'][other_agent_id]
        other_c_knowledge = [str(x) for x in other_c_obs.card_knowledge()[0]]
        probas = self.level_one_belief(self.num_b0_samples, lvl0_counts, c_knowledge, agent_id, other_c_knowledge, other_agent_id, obs, other_obs, other_c_obs)
        lvl1_probs.append(probas)
    
    lvl0_belief = lvl0_counts / np.sum(lvl0_counts, axis=1, keepdims=True)
    lvl1_belief = np.array(lvl1_probs)
    
    return lvl0_belief, lvl1_belief

  def card_knowledge_to_prob_vectors(self, card_knowledges, discards, observed_hands, fireworks, as_counts=False):
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

    card_counts = copy(DEFAULT_CARD_COUNTS)

    discard_inds = []
    for card in discards:
      discard_inds.append((5 * pyhanabi.color_char_to_idx(card['color'])) + card['rank'])
      card_counts[(5 * pyhanabi.color_char_to_idx(card['color'])) + card['rank']] -= 1

    firework_inds = []
    for color, rank in fireworks.items():
      if rank > 0:
        for ind in range(rank):
          firework_inds.append((5 * pyhanabi.color_char_to_idx(color)) + ind)
          card_counts[(5 * pyhanabi.color_char_to_idx(color)) + ind] -= 1

    observed_hand_inds = []
    for hand in observed_hands:
      for card_info in hand:
        if card_info['rank'] < 0:
          break
        observed_hand_inds.append((5 * pyhanabi.color_char_to_idx(card_info['color'])) + card_info['rank'])
        card_counts[(5 * pyhanabi.color_char_to_idx(card_info['color'])) + card_info['rank']] -= 1

    # pad with zeros. also, there shouldn't be any division by zero since the vectors with 0 counts won't be reached
    # due to the fact that the loops only operate over the nonempty infosets.
    prob_vecs = np.zeros((self.n_cards, 25))  
    for set_ind, infoset in enumerate(infosets):
      for ind in infoset:
        if card_counts[ind] > 0:
          prob_vecs[set_ind][ind] = float(card_counts[ind])

      set_sum = np.sum(prob_vecs[set_ind])
      if not as_counts and set_sum > 0:
        prob_vecs[set_ind] /= set_sum

    return prob_vecs

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
    fireworks = my_obs['fireworks']
    prob_vecs = np.zeros(shape=(self.n_cards, 25), dtype=float)
    card_probs = [np.array(prob, dtype=float) / np.sum(prob, dtype=float) for prob in probs]
    attempts = 0
    ind = 0

    if num_samples == 1:
      sample_hand = self.argmax_hand(probs, self_card_knowledge)
      their_observed_hands = self.get_new_player_observed_hands(their_obs, other_c_obs, self_id, sample_hand, other_id)
      new_probs = np.array(self.card_knowledge_to_prob_vectors(other_card_knowledges, discards, their_observed_hands, fireworks, as_counts=False))
      
      return new_probs

    while ind < num_samples and attempts < 100:
      sample_hand = []
      for prob_ind in range(len(self_card_knowledge)):
        card_prob = card_probs[prob_ind]
        sample_hand.append(np.random.choice(25, 1, p=card_prob)[0])
      
      hand_prob = self.hand_prob(probs, self_card_knowledge, sample_hand)

      # don't generate probs for player j if the hand is impossible
      if hand_prob <= 0.:
        attempts += 1
        continue

      their_observed_hands = self.get_new_player_observed_hands(their_obs, other_c_obs, self_id, sample_hand, other_id)

      new_probs = np.array(self.card_knowledge_to_prob_vectors(other_card_knowledges, discards, their_observed_hands, fireworks, as_counts=False))
      new_probs *= hand_prob

      prob_vecs += new_probs
      ind += 1
      
    for prob_ind in range(len(self_card_knowledge)):
       prob_vecs[prob_ind] /= np.sum(prob_vecs[prob_ind])

    return prob_vecs

  @staticmethod
  def hand_prob(prob_counts_in, self_knowledge, hand):
    knowledge_set = set(self_knowledge)
    match_inds = [[] for _ in knowledge_set]
    for ind, knowledge in enumerate(knowledge_set):
      for query_ind, query_knowledge in enumerate(self_knowledge):
        if query_knowledge == knowledge:
          match_inds[ind].append(query_ind)

    prob = 1.
    prob_counts = copy(prob_counts_in)
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
  def argmax_hand(prob_counts_in, self_knowledge):
    knowledge_set = set(self_knowledge)
    match_inds = [[] for _ in knowledge_set]
    for ind, knowledge in enumerate(knowledge_set):
      for query_ind, query_knowledge in enumerate(self_knowledge):
        if query_knowledge == knowledge:
          match_inds[ind].append(query_ind)

    hand = []
    prob_counts = copy(prob_counts_in) + np.random.rand(25) # hack to uniformally sample between multiple argmaxima
    for ind_list in match_inds:
      for ind in ind_list:
        if np.sum(prob_counts[ind_list[0]], dtype=float) == 0.:
          return 0.
        # card_prob = np.array(prob_counts[ind_list[0]], dtype=float) / np.sum(prob_counts[ind_list[0]], dtype=float)
        # card_ind = np.random.choice(25, 1, p=card_prob)[0]
        card_ind = np.argmax(prob_counts[ind_list[0]])
        hand.append(card_ind)
        prob_counts[ind_list[0]][card_ind] = prob_counts[ind_list[0]][card_ind] - 1
        #if prob_counts[ind_list[0]][card_ind] < 0:
        #  return 0.
        #prob *= card_prob[card_ind]

    return hand
