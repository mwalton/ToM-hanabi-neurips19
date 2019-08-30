import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from matrix_game import PartiallyObservableMatrixGame
from tqdm import tqdm

env = PartiallyObservableMatrixGame()

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, obs_sz, hidden_sz, act_sz):
        super(DQN, self).__init__()
        self.hidden = nn.Linear(obs_sz, hidden_sz)
        self.head = nn.Linear(hidden_sz, act_sz)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        return self.head(x)

class DqnAgent(object):
  def __init__(self, obs_sz, hidden_sz, act_sz, mem_cap):
    self.policy_net = DQN(obs_sz, HIDDEN_SIZE, n_actions).to(device)
    self.target_net = DQN(obs_sz, HIDDEN_SIZE, n_actions).to(device)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()
    self.optimizer = optim.Adam(self.policy_net.parameters())
    self.memory = ReplayMemory(10000)
    self.steps_done = 0

  def select_action(self, state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * self.steps_done / EPS_DECAY)
    self.steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

  def optimize_model(self):
    if len(self.memory) < BATCH_SIZE:
        return
    transitions = self.memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    return loss

BATCH_SIZE = 32
HIDDEN_SIZE = 32
GAMMA = 0.9  # .999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200  # 200
TARGET_UPDATE = 10
MEM_CAP = 1000

# Get number of actions from gym action space
obs_sz = env.n_cards
n_actions = env.n_actions

player_1 = DqnAgent(env.n_cards, HIDDEN_SIZE, n_actions, MEM_CAP)
player_2 = DqnAgent(env.n_cards + env.n_actions, HIDDEN_SIZE, n_actions, MEM_CAP)

if __name__ == '__main__':
  num_episodes = 5000
  rewards = []
  losses = []
  for i_episode in tqdm(range(num_episodes)):
      # Initialize the environment and state
      state = env.reset()

      # player 1
      p1_obs = torch.FloatTensor(state[0]).view(1, -1)
      action_1 = player_1.select_action(p1_obs)
      next_state, reward, done = env.step(action_1.item())

      # player 2; observation is card + p1's action
      p2_obs = torch.FloatTensor(np.concatenate((state[1], np.eye(n_actions)[action_1.item()]))).view(1, -1)
      action_2 = player_2.select_action(p2_obs)
      next_state, reward, done = env.step(action_2.item())

      rewards.append(reward)

      player_1.memory.push(p1_obs, action_1, p1_obs, torch.FloatTensor(reward.reshape(1, 1)))
      player_2.memory.push(p2_obs, action_2, p2_obs, torch.FloatTensor(reward.reshape(1, 1)))

      # Perform one step of the optimization (on the target network)
      loss1 = player_1.optimize_model()
      loss2 = player_2.optimize_model()

      if loss1 is not None and loss2 is not None:
        losses.append((loss1.item(), loss2.item()))

      # Update the target network, copying all weights and biases in DQN
      if i_episode % TARGET_UPDATE == 0:
          player_1.target_net.load_state_dict(player_1.policy_net.state_dict())
          player_2.target_net.load_state_dict(player_2.policy_net.state_dict())

  print('Complete')
  plt.plot(rewards)
  plt.show(block=True)
  plt.plot(losses)
  plt.show(block=True)
