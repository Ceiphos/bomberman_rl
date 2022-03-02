import os
import pickle
import random

import numpy as np
#commentar:)

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MODEL_NAME = "linear_model.pt"


class Model:
    def __init__(self, feature_number):
        self.beta = np.zeros((len(ACTIONS), feature_number))

    def predict(self, game_features):
        actionQValues = [self.Q(game_features, i) for i in range(len(ACTIONS))]
        return ACTIONS[np.argmax(actionQValues)]

    def Q(self, game_features, action):
        return game_features @ self.beta[action]

    def updateBeta(self, action, newBeta):
        self.beta[action] = newBeta


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.


    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile(MODEL_NAME):
        self.logger.info("Setting up model from scratch.")
        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()
        self.model = Model(10)
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_NAME, "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        action = self.model.predict(state_to_features(game_state))
        self.logger.debug(f"Querying model for action. Got {action}")
        return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # # For example, you could construct several channels of equal shape, ...
    # channels = []
    # channels.append(...)
    # # concatenate them as a feature tensor (they must have the same shape), ...
    # stacked_channels = np.stack(channels)
    # # and return them as a vector
    # return stacked_channels.reshape(-1)
    return np.ones(10)
