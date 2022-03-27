import numpy as np
import os
import pickle
import sys
sys.path.append(os.getcwd())

import helper

with open("agent_code/regressForest_agent/error_coins.pt", 'rb') as file:
    coins = pickle.load(file)

with open("agent_code/regressForest_agent/error_field.pt", 'rb') as file:
    field = pickle.load(file)

with open("agent_code/regressForest_agent/error_position.pt", 'rb') as file:
    position = pickle.load(file)

helper.findPath(field, (15, 9), position)
print()
