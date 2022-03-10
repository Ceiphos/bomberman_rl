import os
import pickle
import random

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from helper import findPath, epsilonPolicy, findNearestItem, getItemDirection


np.seterr(all='raise')

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']


MODEL_NAME = "regressForest_model.pt"
FEATURE_SIZE = 6


class Model:
    def __init__(self):
        self.forest = RandomForestRegressor(oob_score=True)
        X, Y = self.createRandomData(50)
        self.forest.fit(X, Y)

    def predict(self, game_features):
        actionQValues = self.Q(game_features)
        actions = np.argmax(actionQValues, axis=1).astype(int)
        if (actions.size == 1):
            return ACTIONS[actions[0]]
        else:
            return [ACTIONS[action] for action in actions]

    def Q(self, game_features):
        if (len(game_features.shape) == 1):
            return self.forest.predict(game_features.reshape(1, -1))  # We have only one sample, convert to shape (1, FEATURESIZE)
        else:
            return self.forest.predict(game_features)

    def updateModel(self, game_features, Ys):
        self.forest.fit(game_features, Ys)

    def createRandomData(self, size):
        X = np.random.uniform(size=(size, FEATURE_SIZE))
        Y = np.random.uniform(size=(size, len(ACTIONS)))
        return X, Y


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
        self.model = Model()
        self.eps = epsilonPolicy([0, 1000, 2000], [1, 0.7, 0.3], [1/300, 1/300, 1/200], [0.05]*3)
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
    round = game_state['round']
    if self.train and random.random() < self.eps.epsilon(round):
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS)
    else:
        action = self.model.predict(state_to_features(game_state, self.logger))
        self.logger.debug(f"Querying model for action. Got {action}")
        return action


def state_to_features(game_state: dict, logger) -> np.array:
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

    DIRECTIONS = {
        'UP': np.array((0, -1)),
        'DOWN': np.array((0, 1)),
        'LEFT': np.array((-1, 0)),
        'RIGHT': np.array((1, 0)),
        'NULL': np.array((0, 0))
    }

    features = []

    coins = game_state['coins']
    position = game_state['self'][-1]
    dropped_bomb = game_state['self'][2]
    field = game_state['field']

    # Sourrounding

    # walking direction to nearest coin
    nearest_coin = findNearestItem(field, coins, position)
    features += getItemDirection(field, nearest_coin, position)
    features.append(len(findPath(field, position, nearest_coin)) - 1)

    # Bomb Info:

    # Explosion Map:

    assert len(features) == FEATURE_SIZE
    return np.array(features)
