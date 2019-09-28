import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, combinations, product


class HanabiDeck(object):
    def __init__(self):
        self.n = 50
        self.F = ('r', 'g', 'b', 'w', 'y')
        self.V = (1, 2, 3, 4, 5)
        self.nu = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}
        self.ids, self.counts = self._init_counts()
        self.num_cards = 5

    def _init_counts(self):
        p = []
        ids = []

        for i, c in enumerate(self.F):
            for j, v in enumerate(self.V):
                multiplicity = self.nu[v]
                p.append(multiplicity)
                ids.append(c + str(v))
        return ids, np.array(p)

    def sample(self, num_cards=5):
        p = self.p()
        hand = []
        for _ in range(num_cards):
            hand.append(np.random.choice(self.ids, p=p[:(len(self.F) * len(self.V))]))
        return np.array(hand)

    def p(self):
        p = self.counts / np.sum(self.counts)
        return p

    def meaning_function(self):
        num_hints = (len(self.F) + len(self.V))
        num_cards_in_hand = self.num_cards
        num_unique_cards = len(self.F) * len(self.V)
        M = np.zeros((num_hints, num_unique_cards, num_cards_in_hand))
        M[:, np.where(self.counts > 0)] = 1

        # power_set = self.powerset(range(self.num_cards))
        # color_to_str = {'r': 0, 'g': 1, 'b':2, 'w':3, 'y':4}
        # m = (len(self.F) + len(self.V)) * (2**self.num_cards - 1)
        # n = self.num_cards * len(self.F) * len(self.V)
        # M = np.zeros(shape=(m, n))
        # for c_ind, (label, subset) in enumerate(product(self.F + self.V, power_set)):
        #     print(label, subset)
        #     for ind in subset:
        #         if type(label) == str:
        #             for i in range(5):
        #                 M[c_ind][(25 * ind) + (5 * color_to_str[label]) + i] = 1
        #         else:
        #             offset = 0
        #             for i in self.nu.keys():
        #                 if i == label:
        #                     break
        #                 offset += self.nu[i]
        #             for i in range(5):
        #                 M[c_ind][(25 * ind) + (5 * i) + label - 1] = 1

        return M

    def powerset(self, set_in):
        s = list(set_in)
        res_list = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
        return res_list[1:] if len(res_list) > 0 else []


def pragmatic_agent(M, P, k, type):
    if type == 'listener':
        norm = [0, 1]
    elif type == 'speaker':
        norm = [1, 0]
    else:
        raise ValueError("No such agent type")

    deltas = []

    psi = M * P
    for i in range(1, k + 1):
        new_psi = psi / np.sum(psi, axis=norm[0], keepdims=True)
        new_psi = new_psi / np.sum(new_psi, axis=norm[1], keepdims=True)
        deltas.append(np.sum(np.abs(new_psi - psi), axis=1))
        psi = new_psi

    return psi, deltas


if __name__ == '__main__':
    deck = HanabiDeck()
    # print(np.tile(deck.counts, 5).shape)
    hand = deck.sample()
    # print(deck.powerset(deck.nu.keys()))
    # exit(0)
    print("example hand: {}".format(hand))
    M = deck.meaning_function()
    P = deck.p()
    L, delta_L = pragmatic_agent(M, P, 10, 'listener')
    S, delta_S = pragmatic_agent(M, P, 10, 'speaker')
    plt.plot(delta_L[1:])
    plt.title("Absolute deviation per hint")
    plt.ylabel("Abs diff in model")
    plt.xlabel("Recursion depth")
    plt.show()
    #plt.close()
    plt.plot(delta_S[1:])
    plt.title("Absolute deviation per hint")
    plt.ylabel("Abs diff in model")
    plt.xlabel("Recursion depth")
    plt.show()
    #plt.close()
