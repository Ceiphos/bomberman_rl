import numpy as np
import matplotlib.pyplot as plt
import os

import helper

dir_path = os.path.dirname(os.path.realpath(__file__))
score = np.loadtxt(dir_path + "/logs/score.txt")
rounds = np.arange(len(score)) * 10
epsilon = np.where(rounds < 1000, np.exp(-1/300*rounds) + 0.05, 0.7*np.exp(-1/250*(rounds-1000)))


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')


eps = helper.epsilonPolicy([0, 1000, 2000], [1, 0.7, 0.3], [1/300, 1/300, 1/200], [0.05]*3)
epsilon = [eps.epsilon(round) for round in rounds]

plt.plot(rounds, score, linewidth=0.5)
plt.plot(rounds, epsilon)
avg = movingaverage(score, 10)
plt.plot(rounds[5:-4], movingaverage(score, 10), color="tab:blue")
plt.show()
