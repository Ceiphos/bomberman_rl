import os

import numpy as np
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

round, reward = np.loadtxt(dir_path + "/logs/score.txt", unpack=True)
plt.figure()
plt.plot(round, reward)
plt.grid(linestyle='--')
plt.xlim(None, round[-1])
plt.xlabel('Rounds trained')
plt.ylabel('Total round reward')
plt.show()
