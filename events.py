MOVED_LEFT = 'MOVED_LEFT'
MOVED_RIGHT = 'MOVED_RIGHT'
MOVED_UP = 'MOVED_UP'
MOVED_DOWN = 'MOVED_DOWN'
WAITED = 'WAITED'
INVALID_ACTION = 'INVALID_ACTION'

BOMB_DROPPED = 'BOMB_DROPPED'
BOMB_EXPLODED = 'BOMB_EXPLODED'

CRATE_DESTROYED = 'CRATE_DESTROYED'
COIN_FOUND = 'COIN_FOUND'
COIN_COLLECTED = 'COIN_COLLECTED'

KILLED_OPPONENT = 'KILLED_OPPONENT'
KILLED_SELF = 'KILLED_SELF'

GOT_KILLED = 'GOT_KILLED'
OPPONENT_ELIMINATED = 'OPPONENT_ELIMINATED'
SURVIVED_ROUND = 'SURVIVED_ROUND'

MOVED_CLOSER_TO_COIN = 'MOVED_CLOSER_TO_COIN'
MOVED_CLOSER_TO_CRATE = 'MOVED_CLOSER_TO_CRATE'
OWN_BOMB_CANT_ESCAPE = 'OWN_BOMB_CANT_ESCAPE' #called when agent drops bomb but cant possibly escape from explosion
MOVED_IN_EXPLOSION = 'MOVED_IN_EXPLOSION' #called when agent leaves safe place an gets killed by explosion
BOMB_WILL_DESTROY_CRATE = 'BOMB_WILL_DESTROY_CRATE' #called several times if bomb will destroy mutliple crates
BOMB_THREATS_ENEMY = 'BOMB_THREATS_ENEMY' #called several time if multiple enemies in future explosion field
WAITED_WHILE_NO_BOMB_AROUND = 'WAITED_WHILE_NO_BOMB_AROUND' #called when no bomb on whole field
WAITED_WHILE_IN_DANGER = 'WAITED_WHILE_IN_DANGER' #called if agent in blast coordinates
WAITED_IN_SAFE_SPACE = 'WAITED_IN_SAFE_SPACE' #called if bombs around but agent safe
ESCAPES = 'ESCAPES' #called if agent moves closer to nearest safe tile
MOVED_IN_DEAD_END = 'MOVED_IN_DEAD_END' #called if agent moves in a way that no escape is possible
