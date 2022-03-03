import os
import pickle
import random

import numpy as np

np.seterr(all='raise')

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MODEL_NAME = "linear_model.pt"
FEATURE_SIZE = 34


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
        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()
        self.model = Model(FEATURE_SIZE)
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

    coins = np.array(game_state['coins'])
    position = np.array(game_state['self'][-1])
    field = game_state['field']

    # Sourrounding
    x, y = position + DIRECTIONS['UP']
    up_tile = field[x, y]
    x, y = position + DIRECTIONS['DOWN']
    down_tile = field[x, y]
    x, y = position + DIRECTIONS['LEFT']
    left_tile = field[x, y]
    x, y = position + DIRECTIONS['RIGHT']
    right_tile = field[x, y]
    logger.debug("Calculated sourrounding")

    # 3 nearest coins
    coin_distances = np.linalg.norm(coins-position, axis=1)
    sort_indices = np.argsort(coin_distances)
    nearest_coins = np.ones((3, 2)) * -field.shape[0]  # TODO
    number_of_coins = min(3, len(sort_indices))
    nearest_coins[:number_of_coins] = coins[sort_indices[:number_of_coins]]
    logger.debug(f"Calculated 3 nearest coins. Found {number_of_coins}")

    # walking direction to nearest coin #TODO proper path finding
    nearest_coin_vec = nearest_coins[0]-position
    if (nearest_coin_vec[0] > 0 and right_tile == 0):
        nearest_coin_direction = DIRECTIONS['RIGHT']
    elif (nearest_coin_vec[0] < 0 and left_tile == 0):
        nearest_coin_direction = DIRECTIONS['LEFT']
    elif (nearest_coin_vec[1] > 0 and down_tile == 0):
        nearest_coin_direction = DIRECTIONS['DOWN']
    elif (nearest_coin_vec[1] < 0 and up_tile == 0):
        nearest_coin_direction = DIRECTIONS['UP']
    else:
        nearest_coin_direction = DIRECTIONS['NULL']
    logger.debug("Calculated direction of nearest coin")

    # 10 nearest empty tiles #TODO check accessible using path finding
    empty_tiles = np.array(np.where(field == 0)).transpose()
    empty_tile_distances = np.linalg.norm(empty_tiles-position, axis=1)
    sort_indices = np.argsort(empty_tile_distances)
    nearest_empty_tiles = np.ones((10, 2)) * -1
    number_of_empty_tiles = min(10, len(sort_indices))
    nearest_empty_tiles[:number_of_empty_tiles] = empty_tiles[sort_indices][:number_of_empty_tiles]
    logger.debug(f"Calculated 10 nearest empty tiles. Found {number_of_empty_tiles}")

    # # For example, you could construct several channels of equal shape, ...
    channels = []  # shape n,2
    channels.append(position/field.shape[0])
    channels.append((up_tile, down_tile))
    channels.append((left_tile, right_tile))
    channels.extend(nearest_coins/field.shape[0])
    channels.append(nearest_coin_direction/field.shape[0])
    channels.extend(nearest_empty_tiles/field.shape[0])
    # # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels).reshape(-1)

    assert len(stacked_channels) == FEATURE_SIZE
    return stacked_channels
