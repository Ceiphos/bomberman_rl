from .callbacks import state_to_features, MODEL_NAME, ACTIONS
from .helper import gameStateSymmetry, actionSym, SYMMETRIES
import events as e
from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np
np.seterr(all='raise')


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 100  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
ALPHA = 0.1
GAMMA = 0.9

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    if (old_game_state != None and new_game_state != None):
        for sym in SYMMETRIES:
            features = state_to_features(gameStateSymmetry(old_game_state, sym), self.logger)
            newFeatures = state_to_features(gameStateSymmetry(new_game_state, sym), self.logger)
            self_action = actionSym(self_action, sym)
            self.transitions.append(Transition(features, self_action, newFeatures, reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(Transition(state_to_features(last_game_state, self.logger), last_action, None, reward_from_events(self, events)))

    self.logger.info('Round Ended. Start model update')
    Xs = [[]]*len(ACTIONS)
    Ys = [[]]*len(ACTIONS)
    for (feature, action, next_feature, reward) in self.transitions:
        if type(feature) == type(None) or type(next_feature) == type(None):
            continue
        Qs = [self.model.Q(next_feature, a) for a in range(len(ACTIONS))]
        Y = reward + GAMMA * np.max(Qs)
        Xs[ACTIONS.index(action)].append(feature)
        Ys[ACTIONS.index(action)].append(Y)

    for i, (X, Y) in enumerate(zip(Xs, Ys)):
        self.logger.debug(f'Updated model for action {ACTIONS[i]}')
        self.model.updateModel(i, X, Y)

    # self.transitions.clear()
    self.logger.info('Model update completed')

    # Store the model
    with open(MODEL_NAME, "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 20,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -1.2,#3
        e.KILLED_SELF: -20,
        e.MOVED_UP: 2,
        e.MOVED_DOWN: 2,
        e.MOVED_RIGHT: 2,
        e.MOVED_LEFT: 2,
        e.WAITED: -1.2 #2
    }
    # if invalid_action punished to strong, model will wait. (-5,-3)
    #ifpunished not strong enough, agent acts invalid (-2,-3)
    # (-3,-1) started to wait, but mainly invalid actions, same for (-1,-0.5)
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
