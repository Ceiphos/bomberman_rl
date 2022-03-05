import numpy as np

from callbacks import ACTIONS

SYMMETRIES = ['rotate_right', 'rotate_left', 'rotate_180', 'mirror_x', 'mirror_y', 'id']


def gameStateSymmetry(gameState, symmetry):
    assert symmetry in SYMMETRIES

    if (symmetry == 'id'):
        return gameState

    newGameState = gameState.copy()

    def rotatePositions():
        bombs = gameState['bombs']
        for i, bomb in enumerate(bombs):
            pos = np.array(bomb[0])
            pos = rotation@(pos - center) + center
            bombs[i] = (tuple(pos.astype(int)), bomb[1])

        coins = gameState['coins']
        for i, coin in enumerate(coins):
            pos = np.array(coin)
            pos = rotation@(pos - center) + center
            coins[i] = tuple(pos.astype(int))

        agent = gameState['self']
        pos = np.array(agent[3])
        pos = rotation@(pos - center) + center
        agent = (agent[0], agent[1], agent[2], tuple(pos.astype(int)))

        others = gameState['others']
        for i, other in enumerate(others):
            pos = np.array(other[3])
            pos = rotation@(pos - center) + center
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
    center = (shape-1)/2
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

    field = np.where(field != 0, 1, 0)  # Set all non accesible tiles to 1

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

        DIRECTIONS = (
            np.array((0, -1)),
            np.array((0, 1)),
            np.array((-1, 0)),
            np.array((1, 0)),
        )

        children = []
        for dir in DIRECTIONS:
            pos = current_node.position + dir

            if (field[pos[0], pos[1]] != 0):
                continue

            children.append(Node(current_node, pos))

        for child in children:
            if child in closed_list:
                continue

            child.g = current_node.g + 1
            child.h = np.linalg.norm(child.position - end_node.position)

            if child in open_list:
                index = open_list.index(child)
                if (child.g < open_list[index].g):
                    open_list[index] = child

            else:
                open_list.append(child)


if __name__ == '__main__':
    field = np.zeros((17, 17))
    field[0] = np.ones(17)
    field[-1] = np.ones(17)
    field[:, (0, -1)] = 1
    field[1:7, 5] = 1
    field[3, 2:5] = 1
    start = np.array((1, 1))
    end = np.array((1, 15))

    path = findPath(field, start, end)

    for x, row in enumerate(field):
        for y, value in enumerate(row):
            if (np.any(np.all(np.array((x, y)) == path, axis=1))):
                print("x", end='')
            else:
                print(int(value), end='')
        print()
