import numpy as np
import matplotlib.pyplot as plt

class HanabiDeck(object):
	def __init__(self):
		self.n = 50
		self.F = ('r', 'g', 'b', 'w', 'y')
		self.V = (1, 2, 3, 4, 5)
		self.nu = {1:3, 2:2, 3:2, 4:2, 5:1}
		self.ids, self.counts = self._init_counts()
	
	def _init_counts(self):
		p = []
		ids = []
		
		for i, c in enumerate(self.F):
			for j, v in enumerate(self.V):
				multiplicity = self.nu[v]
				p.append(multiplicity)
				ids.append(c+str(v))
		return ids, np.array(p)
		
		
	def sample(self, num_cards= 5):
		hand = np.random.choice(self.ids, size=num_cards, replace=False, p=self.p())
		return hand
		
	def p(self, context=None):
		''' returns the 'contextual prior' over possible hands
		given context (other player's hand) '''
		if context is None:
			return self.counts / self.n
		else:
			mod_counts = self.counts
			for c in context:
				selected = [i for i, id in enumerate(self.ids) if id == c]
				mod_counts[selected] = mod_counts[selected] - 1
			return mod_counts / (self.n - len(context))
		
	def meaning_function(self, context=[]):
		M = np.zeros((len(self.F) + len(self.V), len(self.ids)))
		for i, h in enumerate(self.F + self.V):
			for j, c in enumerate(self.ids):
				if str(h) in c and not c in context:
					M[i,j] = 1.
		return M
		
def pragmatic_agent(M, P, k, type):
	if type == 'listener':
		norm = [0,1]
	elif type == 'speaker':
		norm = [1,0]
	else:
		raise ValueError("No such agent type")
	
	deltas = []
	
	psi = M * P
	for i in range(1, k+1):
		new_psi = psi / np.sum(psi, axis=norm[0], keepdims=True)
		new_psi = new_psi / np.sum(new_psi, axis=norm[1], keepdims=True)
		deltas.append(np.sum(np.abs(new_psi - psi), axis=1))
		psi = new_psi
		
	return psi, deltas
		
if __name__=='__main__':
	deck = HanabiDeck()
	hand = deck.sample()
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
	plt.close()
	plt.plot(delta_S[1:])
	plt.title("Absolute deviation per hint")
	plt.ylabel("Abs diff in model")
	plt.xlabel("Recursion depth")
	plt.show()
	plt.close()
	
