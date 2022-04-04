import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
sys.path.append(os.getcwd())

import helper

MODEL_NAME = "regressForest_model.pt"

dir_path = os.path.dirname(os.path.realpath(__file__))
rounds, score, rewards, rewards_std, min_rewards, max_rewards = np.loadtxt(dir_path + "/logs/score.txt", unpack=True)

with open("agent_code/regressForest_agent/" + MODEL_NAME, "rb") as file:
    model = pickle.load(file)


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'valid')


eps = helper.epsilonPolicy([0, 2500, 4000], [0.95, 0.45, 0.2], [1 / 1000, 1 / 100, 1 / 1000], [0.05, 0.05, 0.1])
epsilon = [eps.epsilon(round) for round in rounds]

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(rounds, score, linewidth=0.5)
plt.ylim((None, 1))
plt.xlabel('Round trained')
plt.ylabel('OOB score')
plt.grid(linestyle='--')
secax = plt.gca().twinx()
secax.set_ylim(0, 1)
secax.tick_params(axis='y', labelcolor='tab:orange')
secax.plot(rounds, epsilon, color="tab:orange")
secax.set_ylabel(r'$\epsilon$')

plt.subplot(132)
plt.fill_between(rounds, rewards + rewards_std, rewards - rewards_std, alpha=0.5, label='standard deviation')
plt.plot(rounds, rewards, label='Avg round reward')
plt.xlabel('Round trained')
plt.ylabel('Total round reward')
plt.grid(linestyle='--')
plt.legend()
plt.axhline(0, linestyle='--', color='black')

plt.subplot(133)
plt.plot(rounds, min_rewards, linewidth=0.5)
avg = movingaverage(min_rewards, 10)
plt.plot(rounds[5:-4], avg, color="tab:blue", label='Minimal round reward')
plt.plot(rounds, max_rewards, linewidth=0.5, color="tab:green")
avg = movingaverage(max_rewards, 10)
plt.plot(rounds[5:-4], avg, color="tab:green", label='Maximal round reward')
plt.xlabel('Round trained')
plt.ylabel('Total round reward')
plt.legend()
plt.grid(linestyle='--')
plt.axhline(0, linestyle='--', color='black')

plt.tight_layout()
plt.show()
