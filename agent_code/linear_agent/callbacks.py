import os
import pickle
import random

import numpy as np
from helper import findNearestItem, getItemDirection, findPath, addPosition, subPosition, epsilonPolicy

np.seterr(all='raise')

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MODEL_NAME = "linear_model.pt"
FEATURE_SIZE = 10


class Model:
    def __init__(self, feature_number):
        self.beta = np.random.uniform(size=(len(ACTIONS), feature_number))

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
        self.model = Model(FEATURE_SIZE)
        self.eps = epsilonPolicy([0], [0.95], [1 / 100], [0.0])
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
    round = game_state['round']
    if self.train and random.random() < self.eps.epsilon(round):
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
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

    features = []

    coins = game_state['coins']
    position = game_state['self'][-1]
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_field = game_state['explosion_map']
    agents = game_state['others']
    enemy_positions = [x[-1] for x in agents]
    walk_field = field
    bomb_times = [t for _, t in bombs]
    bomb_spots = [pos for pos, _ in bombs]

    crates = []
    walls = []
    for x, column in enumerate(field):
        for y, value in enumerate(column):
            if value == 1:
                crates.append((x, y))
            elif value == -1:
                walls.append((x, y))
    walls.extend(bomb_spots)  # we cant walk into bombs

    # walking direction to nearest coin and path length
    nearest_coin = findNearestItem(walk_field, coins, position)
    features += getItemDirection(walk_field, nearest_coin, position)
    if nearest_coin is not None:
        path = findPath(walk_field, position, nearest_coin)
        # if we stand on top of a coin and drop a bomb path is None
        if path is not None:
            features.append(len(findPath(walk_field, position, nearest_coin)) - 1)
        else:
            features.append(0)

    else:
        features.append(-1)

    # Sourrounding
    features += surrounding(position, walls)  # 4 features
    assert len(features) == FEATURE_SIZE
    return np.array(features)


def surrounding(position, walls):
    directions = (
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
    )
    result = []
    for dir in directions:
        s = addPosition(position, dir)
        if s in walls:
            result += [1]
        else:
            result += [0]

    return result
