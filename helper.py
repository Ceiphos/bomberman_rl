import numpy as np
import time

SYMMETRIES = ['rotate_right', 'rotate_left', 'rotate_180', 'mirror_x', 'mirror_y', 'id']
DIRECTIONS = {
    'UP': [1, 0, 0, 0, 0],
    'DOWN': [0, 1, 0, 0, 0],
    'LEFT': [0, 0, 1, 0, 0],
    'RIGHT': [0, 0, 0, 1, 0],
    'HERE': [0, 0, 0, 0, 1],
    'NULL': [0, 0, 0, 0, 0]
}


def addPosition(a, b):
    return (a[0] + b[0], a[1] + b[1])


def subPosition(a, b):
    return (a[0] - b[0], a[1] - b[1])


def gameStateSymmetry(gameState, symmetry):
    assert symmetry in SYMMETRIES

    if (symmetry == 'id'):
        return gameState

    newGameState = gameState.copy()

    def rotatePositions():
        bombs = gameState['bombs']
        for i, bomb in enumerate(bombs):
            pos = np.array(bomb[0])
            pos = rotation @ (pos - center) + center
            bombs[i] = (tuple(pos.astype(int)), bomb[1])

        coins = gameState['coins']
        for i, coin in enumerate(coins):
            pos = np.array(coin)
            pos = rotation @ (pos - center) + center
            coins[i] = tuple(pos.astype(int))

        agent = gameState['self']
        pos = np.array(agent[3])
        pos = rotation @ (pos - center) + center
        agent = (agent[0], agent[1], agent[2], tuple(pos.astype(int)))

        others = gameState['others']
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
        bombs = gameState['bombs']
        for i, bomb in enumerate(bombs):
            pos = np.array(bomb[0])
            pos[axis] = shape[axis] - 1 - pos[axis]
            bombs[i] = (tuple(pos), bomb[1])

        coins = gameState['coins']
        for i, coin in enumerate(coins):
            pos = np.array(coin)
            pos[axis] = shape[axis] - 1 - pos[axis]
            coins[i] = tuple(pos)

        agent = gameState['self']
        pos = np.array(agent[3])
        pos[axis] = shape[axis] - 1 - pos[axis]
        agent = (agent[0], agent[1], agent[2], tuple(pos))

        others = gameState['others']
        for i, other in enumerate(others):
            pos = np.array(other[3])
            pos[axis] = shape[axis] - pos[axis]
            other = (other[0], other[1], other[2], tuple(pos))
            others[i] = other

        newGameState['bombs'] = bombs
        newGameState['coins'] = coins
        newGameState['self'] = agent
        newGameState['others'] = others

    shape = np.array(gameState['field'].shape)
    center = (shape - 1) / 2
    if (symmetry == 'rotate_right'):
        newGameState['field'] = np.rot90(gameState['field'], -1)
        newGameState['explosion_map'] = np.rot90(gameState['explosion_map'], -1)

        rotation = np.array(((0, 1), (-1, 0)))
        rotatePositions()

    if (symmetry == 'rotate_left'):
        newGameState['field'] = np.rot90(gameState['field'], 1)
        newGameState['explosion_map'] = np.rot90(gameState['explosion_map'], 1)

        rotation = np.array(((0, -1), (1, 0)))
        rotatePositions()

    if (symmetry == 'mirror_x'):
        newGameState['field'] = np.fliplr(gameState['field'])
        newGameState['explosion_map'] = np.fliplr(gameState['explosion_map'])

        flipPosition(0)

    if (symmetry == 'mirror_y'):
        newGameState['field'] = np.flipud(gameState['field'])
        newGameState['explosion_map'] = np.flipud(gameState['explosion_map'])

        flipPosition(1)

    if (symmetry == 'rotate_180'):
        newGameState['field'] = np.rot90(gameState['field'], 2)
        newGameState['explosion_map'] = np.rot90(gameState['explosion_map'], 2)

        rotation = np.array(((-1, 0), (0, -1)))
        rotatePositions()

    return newGameState


def actionSym(action, symmetry):
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
    """ 
    Calculate path from start to end in field as a list of postions

    Uses A* algortihm
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
    nearestItems = findNNearestItems(field, items, position, 1)
    if nearestItems is not None:
        return nearestItems[0]
    else:
        return None


class epsilonPolicy:
    def __init__(self, rounds, starting_eps, lambdas, min_eps):
        assert len(rounds) == len(starting_eps) == len(lambdas) == len(min_eps)
        assert rounds[0] == 0
        self.rounds = rounds
        self.starting_eps = starting_eps
        self.lambdas = lambdas
        self.min_eps = min_eps

    def epsilon(self, round):
        for i, threshold in enumerate(self.rounds):
            if (i == len(self.rounds) - 1):
                return self.starting_eps[i] * np.exp(-self.lambdas[i] * (round - threshold)) + self.min_eps[i]
            elif threshold <= round < self.rounds[i + 1]:
                return self.starting_eps[i] * np.exp(-self.lambdas[i] * (round - threshold)) + self.min_eps[i]

def check_own_escape(field, position):
    vector_to_safe = [(4,0),
                      (-4,0),
                      (0,4),
                      (0,-4),
                      (1,1),
                      (1,-1),
                      (-1,1),
                      (-1,-1),
                      (2,1),
                      (2,-1),
                      (-2,1),
                      (-2,-1),
                      (1,2),
                      (-1,2),
                      (1,-2),
                      (-1,-2)]
    for vec in vector_to_safe:
        check_position = addPosition(position, vec)
        if (check_position[0] in range(17) and check_position[1] in range(17)):
            path = findPath(field, position, check_position)
            if (path != None and len(path)<5):
                return True
    return False

def dangerous_position(position, bombs):
    #score between 0 and 3, 0 for no danger, 3 for death in next step 
    danger_score = 0
    directions = (
        (0, -1),(0, -2),(0, -3),
        (0, 1),(0, 2),(0, 3),
        (-1, 0),(-2, 0),(-3, 0),
        (1, 0),(2, 0),(3, 0)
    )

    for (pos,t) in bombs:
        for dir in directions:
            danger_coord = addPosition(pos,dir)
            if (danger_coord == position and (3-t)> danger_score):
                danger_score = (3-t)
                continue
    return danger_score
        
def find_next_to_crate(field):
    #return possible positions next to crates as tuples
    next_to_crates = []
    ind_crates = np.argwhere(field == 1)
    for x, y in ind_crates:
        for u in [-1,1]:
            if field[x+u,y]==0:
                next_to_crates.append((x+u,y))
        for v in[-1,1]:
            if field[x,y+v]==0:
                next_to_crates.append((x,y+v))
    return next_to_crates
        
        


if __name__ == '__main__':
    field = np.zeros((17, 17))
    field[0] = np.ones(17)
    field[-1] = np.ones(17)
    field[:, (0, -1)] = 1
    field[1:15, 5] = 1
    field[3, 2:7] = 1
    field[1:4, 7] = 1
    start = (1, 1)
    end = (1, 15)
    path = findPath(field, start, end)
    print(path)
    print(getItemDirection(field, end, start))
    items = [(3, 8), (1, 6), (14, 7), (8, 3)]
    print(findNNearestItems(field, items, start, 2))
    print(findNearestItem(field, [(1, 6)], start))

    for y, row in enumerate(field.T):
        for x, value in enumerate(row):
            if ((x, y) in path):
                print("x", end='')
            elif ((x, y) in items):
                print("I", end='')
            else:
                print(int(value), end='')
        print()

    a = epsilonPolicy([0, 1000, 2000], [1, 0.7, 0.3], [1 / 500, 1 / 300, 1 / 100], [0.05] * 3)
    print(a.epsilon(0))
    print(a.epsilon(999))
    print(a.epsilon(1000))
    print(a.epsilon(1999))
    print(a.epsilon(2000))
    print(a.epsilon(3000))
