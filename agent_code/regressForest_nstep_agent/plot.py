import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys
sys.path.append(os.getcwd())

import helper
AGENT_NAME = "regressForest_nstep"
MODEL_NAME = AGENT_NAME + "_model.pt"

dir_path = os.path.dirname(os.path.realpath(__file__))
rounds, score, rewards, rewards_std, min_rewards, max_rewards = np.loadtxt(dir_path + "/logs/score.txt", unpack=True)
epsilon = np.where(rounds < 1000, np.exp(-1 / 300 * rounds) + 0.05, 0.7 * np.exp(-1 / 250 * (rounds - 1000)))

with open(f"agent_code/{AGENT_NAME}_agent/" + MODEL_NAME, "rb") as file:
    model = pickle.load(file)


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'valid')


eps = helper.epsilonPolicy([0, 200, 1000, 2000], [1, 1, 0.7, 0.3], [0, 1 / 300, 1 / 300, 1 / 200], [0.05] * 4)
epsilon = [eps.epsilon(round) for round in rounds]

plt.figure(figsize=(12, 5))
plt.subplot(131)
plt.plot(rounds, score)
plt.ylim((None, 1))
secax = plt.gca().twinx()
secax.set_ylim(0, 1)
secax.tick_params(axis='y', labelcolor='tab:orange')
secax.plot(rounds, epsilon, color="tab:orange")

plt.subplot(132)
plt.fill_between(rounds, rewards + rewards_std, rewards - rewards_std, alpha=0.5)
plt.plot(rounds, rewards)
plt.grid(linestyle='--')
plt.ylim(np.amin(rewards) * 1.5, max(100, np.amax(rewards) * 3))
print(np.amax(rewards))
plt.axhline(0, linestyle='--', color='black')

plt.subplot(133)
plt.plot(rounds, min_rewards, linewidth=0.5)
avg = movingaverage(min_rewards, 10)
plt.plot(rounds[5:-4], avg, color="tab:blue")
plt.plot(rounds, max_rewards, linewidth=0.5, color="tab:green")
avg = movingaverage(max_rewards, 10)
plt.plot(rounds[5:-4], avg, color="tab:green")
plt.axhline(0, linestyle='--', color='black')

plt.tight_layout()
plt.show()
