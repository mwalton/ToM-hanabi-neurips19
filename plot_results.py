from agents.rainbow.third_party.dopamine import logger
from agents.rainbow.third_party.dopamine.colab import utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    dat = utils.read_experiment('./')
    import matplotlib
    print(matplotlib.get_backend())
    plt.plot(np.linspace(0., 1.))
    plt.show()
    print('Done')
