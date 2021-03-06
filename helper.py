import numpy as np
import time

import copy

SYMMETRIES = ['id', 'rotate_right', 'rotate_left', 'rotate_180', 'mirror_x', 'mirror_y']
DIRECTIONS = {
    'UP': [1, 0, 0, 0, 0],
    'DOWN': [0, 1, 0, 0, 0],
    'LEFT': [0, 0, 1, 0, 0],
    'RIGHT': [0, 0, 0, 1, 0],
    'HERE': [0, 0, 0, 0, 1],
    'NULL': [0, 0, 0, 0, 0]
}


def addPosition(a, b):
    """Adds tuples a and b elementwise and returns a new tuple with the result

    Parameters
    ----------
    a : tuple (2dim)
    b : tuple (2dim)

    Returns
    -------
    tuple (2dim)
    """
    return (a[0] + b[0], a[1] + b[1])


def subPosition(a, b):
    """Subtracts tuples a and b elementwise and returns a new tuple with the result

    Parameters
    ----------
    a : tuple (2dim)
    b : tuple (2dim)

    Returns
    -------
    tuple (2dim)
    """
    return (a[0] - b[0], a[1] - b[1])


def gameStateSymmetry(gameState, symmetry):
    """Performs the symmetry transformation on the gameState and returns the transformed state

    Parameters
    ----------
    gameState : dic
    symmetry : str

    Returns
    -------
    newGameState : dic
        The resulting game state after the symmetry was applied
    """
    assert symmetry in SYMMETRIES

    if (symmetry == 'id'):
        return gameState

    newGameState = copy.deepcopy(gameState)

    def rotatePositions():
        bombs = newGameState['bombs']
        for i, bomb in enumerate(bombs):
            pos = np.array(bomb[0])
            pos = rotation @ (pos - center) + center
            bombs[i] = (tuple(pos.astype(int)), bomb[1])

        coins = newGameState['coins']
        for i, coin in enumerate(coins):
            pos = np.array(coin)
            pos = rotation @ (pos - center) + center
            coins[i] = tuple(pos.astype(int))

        agent = newGameState['self']
        pos = np.array(agent[3])
        pos = rotation @ (pos - center) + center
        agent = (agent[0], agent[1], agent[2], tuple(pos.astype(int)))

        others = newGameState['others']
        for i, other in enumerate(others):
            pos = np.array(other[3])
            pos = rotation @ (pos - center) + center
            other = (other[0], other[1], other[2], tuple(pos.astype(int)))
            others[i] = other

        newGameState['bombs'] = bombs
        newGameState['coins'] = coins
        newGameState['self'] = agent
        newGameState['others'] = others

    def flipPosition(axis):
        bombs = newGameState['bombs']
        for i, bomb in enumerate(bombs):
            pos = np.array(bomb[0])
            pos[axis] = shape[axis] - 1 - pos[axis]
            bombs[i] = (tuple(pos), bomb[1])

        coins = newGameState['coins']
        for i, coin in enumerate(coins):
            pos = np.array(coin)
            pos[axis] = shape[axis] - 1 - pos[axis]
            coins[i] = tuple(pos)

        agent = newGameState['self']
        pos = np.array(agent[3])
        pos[axis] = shape[axis] - 1 - pos[axis]
        agent = (agent[0], agent[1], agent[2], tuple(pos))

        others = newGameState['others']
        for i, other in enumerate(others):
            pos = np.array(other[3])
            pos[axis] = shape[axis] - pos[axis]
            other = (other[0], other[1], other[2], tuple(pos))
            others[i] = other

        newGameState['bombs'] = bombs
        newGameState['coins'] = coins
        newGameState['self'] = agent
        newGameState['others'] = others

    shape = np.array(newGameState['field'].shape)
    center = (shape - 1) / 2
    if (symmetry == 'rotate_right'):
        newGameState['field'] = np.rot90(newGameState['field'], -1)
        newGameState['explosion_map'] = np.rot90(newGameState['explosion_map'], -1)

        rotation = np.array(((0, 1), (-1, 0)))
        rotatePositions()

    if (symmetry == 'rotate_left'):
        newGameState['field'] = np.rot90(newGameState['field'], 1)
        newGameState['explosion_map'] = np.rot90(newGameState['explosion_map'], 1)

        rotation = np.array(((0, -1), (1, 0)))
        rotatePositions()

    if (symmetry == 'mirror_x'):
        newGameState['field'] = np.fliplr(newGameState['field'])
        newGameState['explosion_map'] = np.fliplr(newGameState['explosion_map'])

        flipPosition(0)

    if (symmetry == 'mirror_y'):
        newGameState['field'] = np.flipud(newGameState['field'])
        newGameState['explosion_map'] = np.flipud(newGameState['explosion_map'])

        flipPosition(1)

    if (symmetry == 'rotate_180'):
        newGameState['field'] = np.rot90(newGameState['field'], 2)
        newGameState['explosion_map'] = np.rot90(newGameState['explosion_map'], 2)

        rotation = np.array(((-1, 0), (0, -1)))
        rotatePositions()

    return newGameState


def actionSym(action, symmetry):
    """Performs the symmetry transformation on the action and returns the transformed action

    Parameters
    ----------
    action : str
    symmetry : str

    Returns
    -------
    symAction : str
        The resulting action after the symmetry was applied
    """
    assert symmetry in SYMMETRIES
    symAction = ['UP', 'RIGHT', 'DOWN', 'LEFT']

    if (symmetry == 'id' or action not in symAction):
        return action

    index = symAction.index(action)
    newAction = action
    if (symmetry == 'rotate_left'):
        newAction = symAction[(index - 1) % len(symAction)]
    if (symmetry == 'rotate_right'):
        newAction = symAction[(index + 1) % len(symAction)]
    if (symmetry == 'rotate_180'):
        newAction = symAction[(index + 2) % len(symAction)]
    if (symmetry == 'mirror_x'):
        if (action == 'RIGHT' or action == 'LEFT'):
            newAction = symAction[(index + 2) % len(symAction)]
    if (symmetry == 'mirror_y'):
        if (action == 'DOWN' or action == 'UP'):
            newAction = symAction[(index + 2) % len(symAction)]

    return newAction


def findPath(field, start, end):
    """Calculate path from start to end in field as a list of postions

    Uses A* algortihm

    Parameters
    ----------
    field : 2dim ndarray
    start : 2dim tuple
    end : 2dim tuple

    Retruns
    -------
    path : List of 2dim tuples
        path contains the positions along the calculated path, including start and end
    """
    assert start is not None
    assert end is not None
    # Check if endpoint is accesible at all
    if field[end[0], end[1]] != 0:
        return None

    class Node():
        def __init__(self, parent=None, position=None):
            self.parent = parent
            self.position = position

            self.g = 0
            self.h = 0

        @property
        def f(self):
            return self.g + self.h

        def __eq__(self, other):
            return np.all(self.position == other.position)

    start_node = Node(None, start)
    end_node = Node(None, end)

    open_list = []
    closed_list = []

    open_list.append(start_node)

    while len(open_list) > 0:
        current_node = open_list[0]
        current_index = 0
        for i, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = i

        open_list.pop(current_index)
        closed_list.append(current_node)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        directions = (
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
        )

        children = []
        for dir in directions:
            pos = addPosition(current_node.position, dir)

            if (field[pos[0], pos[1]] != 0):
                continue

            children.append(Node(current_node, pos))

        for child in children:
            if child in closed_list:
                continue

            child.g = current_node.g + 1
            child.h = np.linalg.norm(subPosition(child.position, end_node.position))

            if child in open_list:
                index = open_list.index(child)
                if (child.g < open_list[index].g):
                    open_list[index] = child

            else:
                open_list.append(child)


def getItemDirection(field, item, position):
    """Calculates the direction of the item from the curren position

    Performs a path finding to the item and calculates the direction from the first step

    Parameters
    ----------
    field : 2dim ndarray
    item : 2dim tuple
    position : 2dim tuple

    Returns
    direction : List
        The one-hot encoded direction to the item
    """
    assert position is not None
    if item is None:
        return DIRECTIONS['NULL']

    path = findPath(field, position, item)
    if path is not None:
        if (len(path) > 1):
            x, y = subPosition(path[1], position)
            if (x == 1 and y == 0):
                return DIRECTIONS['RIGHT']
            elif (x == -1 and y == 0):
                return DIRECTIONS['LEFT']
            elif (y == 1 and x == 0):
                return DIRECTIONS['UP']
            elif (y == -1 and x == 0):
                return DIRECTIONS['DOWN']
            else:
                raise RuntimeError(f"Calculated direction is not valid: {x},{y}")
        else:
            return DIRECTIONS['HERE']
    else:
        return DIRECTIONS['NULL']


def findNNearestItems(field, items, position, n):
    """Performs a breadth first search to find the n closest members of items

    Parameters
    ----------
    field : 2dim ndarray
    items : List of 2dim tuples
    position : 2dim tuple
    n : int

    Returns
    -------
    nearestItems : List of 2dim tuples
    """
    if (len(items) == 0):
        return None
    visited = []
    queue = [position]

    directions = (
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
    )

    result = []

    while len(queue) > 0:
        node = queue.pop(0)
        if (node in items):
            result.append(node)
            if len(result) == n:
                return result
        visited.append(node)
        for dir in directions:
            next_node = addPosition(node, dir)
            if (next_node in visited or field[next_node[0], next_node[1]] != 0 or next_node in queue):
                continue
            else:
                queue.append(next_node)


def findNearestItem(field, items, position):
    """Finds the closest members of items using a call to findNNearestItems

    Parameters
    ----------
    field : 2dim ndarray
    items : List of 2dim tuples
    position : 2dim tuple

    Returns
    -------
    nearestItem : 2dim tuple
    """
    nearestItems = findNNearestItems(field, items, position, 1)
    if nearestItems is not None:
        return nearestItems[0]
    else:
        return None


class epsilonPolicy:
    """Simple interface to provide epsilon values as a function of the round

    The epsilons are calculated using an exponential decay with specified parameters
    Multiple decays starting at different rounds can be specified

    Parameters
    ----------
    rounds : List of int
        The rounds where the different decays start, must be 0 in the 0th entry
    starting_eps : List of float
        The epsilon values to start the decay, the corresponding min_eps is added
    lambdas : List of float
        The decay speeds
    min_eps : List of float
        The minimal value each decay converges to
    """

    def __init__(self, rounds, starting_eps, lambdas, min_eps):
        assert len(rounds) == len(starting_eps) == len(lambdas) == len(min_eps)
        assert rounds[0] == 0
        self.rounds = rounds
        self.starting_eps = starting_eps
        self.lambdas = lambdas
        self.min_eps = min_eps

    def epsilon(self, round):
        """Returns the epsilon value for the given round

        This is calculated by eps = eps_0 * exp(-(round-round_0)*lambda) + eps_min
        where eps_0, lambda, round_0 and eps_min are the corresponding parameters accroding to the rounds list

        Parameters
        ----------
        round : int

        Returns
        -------
        epsilon : float
        """
        for i, threshold in enumerate(self.rounds):
            if (i == len(self.rounds) - 1):
                return self.starting_eps[i] * np.exp(-self.lambdas[i] * (round - threshold)) + self.min_eps[i]
            elif threshold <= round < self.rounds[i + 1]:
                return self.starting_eps[i] * np.exp(-self.lambdas[i] * (round - threshold)) + self.min_eps[i]


def check_own_escape(field, position):
    """Checks wheter a safe spot can be reached from the current position if a bomb is dropped

    Iterates to all possible safe spots and returns true if one can be reached by a path  in the next 4 steps

    Parameters
    ----------
    field : 2dim ndarayy
    position : 2dim tuple

    Returns
    -------
    safe : bool
    """
    vector_to_safe = [(4, 0),
                      (-4, 0),
                      (0, 4),
                      (0, -4),
                      (1, 1),
                      (1, -1),
                      (-1, 1),
                      (-1, -1),
                      (2, 1),
                      (2, -1),
                      (-2, 1),
                      (-2, -1),
                      (1, 2),
                      (-1, 2),
                      (1, -2),
                      (-1, -2)]
    # Iterate through all possible safe spots and check wheter a short enough path exists
    for vec in vector_to_safe:
        check_position = addPosition(position, vec)
        if (check_position[0] in range(17) and check_position[1] in range(17)):
            path = findPath(field, position, check_position)
            if (path != None and len(path) <= 5):
                return True
    else:
        return False


def dangerous_position(position, bombs):
    """Calculates a danger score for the position by the given bombs

    The score is between 0 and 4 with 0 meaning no danger and 4 death in the next step

    Parameters
    ----------
    position : 2dim tuple
    bombs: List of 2dim tuple

    Returns
    -------
    in_danger : bool
    danger_score : int
    """
    # score between 0 and 4, 0 for no danger, 4 for death in next step
    danger_score = 0
    directions = (
        (0, -1), (0, -2), (0, -3),
        (0, 1), (0, 2), (0, 3),
        (-1, 0), (-2, 0), (-3, 0),
        (1, 0), (2, 0), (3, 0)
    )
    in_danger = False
    for (pos, t) in bombs:
        if pos == position:
            in_danger = True
            danger_score = max(danger_score, 4 - t)  # t is between 0 and 3
        for dir in directions:
            danger_coord = addPosition(pos, dir)
            if (danger_coord == position):
                danger_score = max(danger_score, 4 - t)
                in_danger = True
                continue
    return in_danger, danger_score


def future_explosion_field(bomb, field):
    """Calculates the positions that will contain a explosion in the future of the given bomb

    Parameters
    ----------
    bomb : 2dim tuple
    field: 2dim ndarray

    Returns
    -------
    future_explosions : List of 2dim tuple
    """
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    future_explosions = [bomb]
    for dir in directions:
        check = bomb
        for i in range(3):
            check = addPosition(check, dir)
            if (check[0] in range(17) and check[1] in range(17)):
                if field[check] == -1:
                    break  # Explosion blocked by a wall, check next direction
                else:
                    future_explosions.append(check)
    return future_explosions


def find_next_to_crate(field):
    """Returns all possible positions that are next to crates

    Parameters
    ----------
    field : 2dim ndarray

    Returns
    -------
    next_to_crates : List of 2dim tuple

    """
    next_to_crates = []
    ind_crates = np.argwhere(field == 1)
    for x, y in ind_crates:
        for u in [-1, 1]:
            if field[x + u, y] == 0:
                next_to_crates.append((x + u, y))
        for v in [-1, 1]:
            if field[x, y + v] == 0:
                next_to_crates.append((x, y + v))
    return next_to_crates


if __name__ == '__main__':
    field = np.zeros((17, 17))
    field[0] = np.ones(17)
    field[-1] = np.ones(17)
    field[:, (0, -1)] = 1
    field[1:15, 7] = 1
    field[3, 2:7] = 1
    field[1:4, 7] = 1
    start = (1, 1)
    end = addPosition(start, (0, 4))
    path = findPath(field, start, end)
    print(path)
    print(getItemDirection(field, end, start))
    items = [(3, 8), (1, 6), (14, 7), (8, 3)]
    print(findNNearestItems(field, items, start, 2))
    print(findNearestItem(field, [(1, 6)], start))

    game_state = {
        'field': field,
        'explosion_map': np.zeros_like(field),
        'bombs': [],
        'self': [0, 1, 2, [1, 1]],
        'others': [],
        'coins': items,
    }

    new_game_state = gameStateSymmetry(game_state, 'rotate_left')
    # new_game_state = gameStateSymmetry(new_game_state, 'mirror_y')
    # new_game_state = gameStateSymmetry(new_game_state, 'rotate_180')
    print(np.all(game_state['field'] == new_game_state['field']))
    print(game_state['coins'] == new_game_state['coins'])
    action = 'RIGHT'
    print(actionSym(action, 'rotate_left'))

    for y, row in enumerate(field.T):
        for x, value in enumerate(row):
            if ((x, y) in path):
                print("x", end='')
            elif ((x, y) in items):
                print("I", end='')
            else:
                print(int(value), end='')
        print()

    print()
    for y, row in enumerate(new_game_state['field'].T):
        for x, value in enumerate(row):
            if ((x, y) in new_game_state['bombs']):
                print("x", end='')
            elif ((x, y) in new_game_state['coins']):
                print("I", end='')
            else:
                print(int(value), end='')
        print()
