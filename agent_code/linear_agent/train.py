from .callbacks import state_to_features, MODEL_NAME, ACTIONS
import events as e
from collections import namedtuple, deque

from helper import gameStateSymmetry, actionSym, SYMMETRIES

import pickle
import os
import random
from typing import List
import numpy as np
np.seterr(all='raise')


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 500000  # keep only ... last transitions
TRAIN_BATCH_SIZE = 5000
ALPHA = 0.1
GAMMA = 0.9


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.current_round_reward = 0
    self.round = 0

    if os.path.exists("logs/score.txt"):
        os.remove("logs/score.txt")


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
    reward = reward_from_events(self, events)
    self.current_round_reward += reward
    if (old_game_state != None and new_game_state != None):
        features = state_to_features(old_game_state, self.logger)
        newFeatures = state_to_features(new_game_state, self.logger)
        self.transitions.append(Transition(features, self_action, newFeatures, reward))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    reward = reward_from_events(self, events)
    self.current_round_reward += reward
    self.round += 1
    features = state_to_features(last_game_state, self.logger)
    self.transitions.append(Transition(features, last_action, None, reward))

    if (len(self.transitions) < TRAIN_BATCH_SIZE):
        return

    self.logger.info('Round Ended. Start model update')

    batch = random.sample(self.transitions, TRAIN_BATCH_SIZE)
    oldBeta = self.model.beta
    for i, train_action in enumerate(ACTIONS):
        sum = 0
        n = 0
        for (feature, action, next_feature, reward) in batch:
            if type(feature) == type(None) or type(next_feature) == type(None) or action != train_action:
                continue
            Qs = [self.model.Q(next_feature, a) for a in range(len(ACTIONS))]
            Y = reward + GAMMA * np.max(Qs)
            sum += np.transpose(feature) * (Y - feature @ oldBeta[i])
            n += 1

        if n != 0:
            newBeta = oldBeta[i] + ALPHA / n * sum
            self.logger.debug(f'Updated beta for action {ACTIONS[i]}')
            self.model.updateBeta(i, newBeta)

    with open("logs/score.txt", "a") as file:
        log_string = ""
        log_string += f" {self.round}"
        log_string += f" {self.current_round_reward}"
        file.write(log_string + "\n")

    self.current_round_reward = 0

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
        e.COIN_COLLECTED: 5,
        e.MOVED_CLOSER_TO_COIN: 1.5,
        # e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -5,
        # e.KILLED_SELF: -5,
        e.MOVED_UP: -0.5,
        e.MOVED_DOWN: -0.5,
        e.MOVED_RIGHT: -0.5,
        e.MOVED_LEFT: -0.5,
        e.WAITED: -1,
        e.BOMB_DROPPED: -1,
        # For Scenario 2+
        e.BOMB_THREATS_ENEMY: 10,
        e.BOMB_WILL_DESTROY_CRATE: 0.5,
        e.MOVED_IN_EXPLOSION: -5,
        e.ESCAPES: 1.5,
        e.OWN_BOMB_CANT_ESCAPE: -10,
        e.WAITED_WHILE_NO_BOMB_AROUND: -5,
        e.WAITED_WHILE_IN_DANGER: -7
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
