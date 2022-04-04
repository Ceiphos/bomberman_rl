import os
import pickle
import random

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from helper import findPath, epsilonPolicy, findNearestItem, getItemDirection, addPosition, subPosition, DIRECTIONS, dangerous_position, find_next_to_crate, check_own_escape, future_explosion_field


np.seterr(all='raise')

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


MODEL_NAME = "regressForest_model.pt"
FEATURE_SIZE = 34
escape_path = []


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
        self.model = Model()
        self.eps = epsilonPolicy([0, 2500], [0.95, 0.45], [1 / 1000, 1 / 100], [0.05, 0.05])
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_NAME, "rb") as file:
            self.model = pickle.load(file)
            assert self.model.forest.n_features_in_ == FEATURE_SIZE, "Feature size of loaded model does not match"


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

    # For Coin-Heaven scenario
    # features = []

    # coins = game_state['coins']
    # position = game_state['self'][-1]
    # field = game_state['field']
    # bombs = game_state['bombs']
    # explosion_field = game_state['explosion_map']
    # agents = game_state['others']
    # enemy_positions = [x[-1] for x in agents]
    # walk_field = field
    # bomb_times = [t for _, t in bombs]
    # bomb_spots = [pos for pos, _ in bombs]

    # crates = []
    # walls = []
    # for x, column in enumerate(field):
    #     for y, value in enumerate(column):
    #         if value == 1:
    #             crates.append((x, y))
    #         elif value == -1:
    #             walls.append((x, y))
    # walls.extend(bomb_spots)  # we cant walk into bombs

    # # walking direction to nearest coin and path length
    # nearest_coin = findNearestItem(walk_field, coins, position)
    # features += getItemDirection(walk_field, nearest_coin, position)
    # if nearest_coin is not None:
    #     path = findPath(walk_field, position, nearest_coin)
    #     # if we stand on top of a coin and drop a bomb path is None
    #     if path is not None:
    #         features.append(len(findPath(walk_field, position, nearest_coin)) - 1)
    #     else:
    #         features.append(0)

    # else:
    #     features.append(-1)

    # # Sourrounding
    # features += surrounding(position, walls)  # 4 features
    # assert len(features) == FEATURE_SIZE
    # return np.array(features)

    features = []

    coins = game_state['coins']
    position = game_state['self'][-1]
    bomb_possible = game_state['self'][-2]
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_field = game_state['explosion_map']
    agents = game_state['others']
    enemy_positions = [x[-1] for x in agents]
    walk_field = field

    # Bomb Info:
    bomb_times = [t for _, t in bombs]
    bomb_spots = [pos for pos, _ in bombs]
    bomb_field = np.ones_like(field) * 10  # Initialize the bomb field with a large value, to distiguish from a bomb
    for (x, y), t in bombs:
        walk_field[x, y] = -2  # treat bombs like walls, because we cant walk through them
        single_bomb_field = future_explosion_field((x, y), field)
        for tile in single_bomb_field:
            bomb_field[tile] = min(t, bomb_field[tile])

    if bomb_field[position[0], position[1]] < 10:  # if in bomb field, 1, else 0
        features.append(1)
    else:
        features.append(0)
    if bomb_possible:
        features.append(1)  # if can throw bomb 1, else 0
    else:
        features.append(0)

    # bomb power (how many crates and enemies would be endangered if bomb dropped)
    tiles_to_check = future_explosion_field(position, field)
    destroyable_crates = 0
    threatened_enemy = 0
    for tile in tiles_to_check:
        if field[tile] == 1:
            destroyable_crates += 1
        elif tile in enemy_positions:
            threatened_enemy += 1
    features.append(destroyable_crates)
    features.append(threatened_enemy)

    danger_score_position = bomb_field[position]
    features.append(danger_score_position)

    crates = []
    walls = []
    for x, column in enumerate(field):
        for y, value in enumerate(column):
            if value == 1:
                crates.append((x, y))
            elif value == -1:
                walls.append((x, y))
    walls.extend(bomb_spots)  # we cant walk into bombs

    # walking direction to nearest coin
    nearest_coin = findNearestItem(walk_field, coins, position)
    features += getItemDirection(walk_field, nearest_coin, position)
    if nearest_coin is not None:
        path = findPath(walk_field, position, nearest_coin)
        if path is not None:
            features.append(len(findPath(walk_field, position, nearest_coin)) - 1)
        else:
            with open("error_field.pt", "wb") as file:
                pickle.dump(walk_field, file)
            with open("error_coins.pt", "wb") as file:
                pickle.dump(coins, file)
            with open("error_position.pt", "wb") as file:
                pickle.dump(position, file)
            features.append(-1)
    else:
        features.append(-1)

    # Crates:
    next_to_crates = find_next_to_crate(walk_field)
    nearest_crate = findNearestItem(walk_field, next_to_crates, position)
    features += getItemDirection(walk_field, nearest_crate, position)

    # Explosion Map:
    explosion_field = game_state['explosion_map']
    danger_field = []
    # We treat explosions as walls, as we don't want to walk into either #TODO Check if makes sense
    for x, column in enumerate(explosion_field):
        for y, value in enumerate(column):
            if value != 0:
                walls.append((x, y))
                danger_field.append((x, y))

    # Escape: esc=1 if escape possible, 0 else. direction to nearest safe place (no current explosion and no threat of bombs)
    esc = check_own_escape(walk_field, position)
    if esc:
        features.append(1)
    else:
        features.append(0)
    for (bomb_pos, t) in bombs:
        danger_field.append(future_explosion_field(bomb_pos, field))
    free_tiles = np.argwhere(walk_field == 0)
    safe_tiles = []
    for [x, y] in free_tiles:
        if (x, y) not in danger_field:
            safe_tiles.append((x, y))
    nearest_safe_tile = findNearestItem(walk_field, safe_tiles, position)
    features += getItemDirection(walk_field, nearest_safe_tile, position)

    # Sourrounding
    features += surrounding(position, walls, crates, enemy_positions)  # 12 features
    assert len(features) == FEATURE_SIZE
    return np.array(features)

# For Coin-Heaven scenario
# def surrounding(position, walls):
#     directions = (
#         (0, -1),
#         (0, 1),
#         (-1, 0),
#         (1, 0),
#     )
#     result = []
#     for dir in directions:
#         s = addPosition(position, dir)
#         if s in walls:
#             result += [1]
#         else:
#             result += [0]

#     return result


def surrounding(position, walls, crates, agents):
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
            result += [1, 0, 0]
        elif s in crates:
            result += [0, 1, 0]
        elif s in agents:
            result += [0, 0, 1]
        else:
            result += [0, 0, 0]

    return result
