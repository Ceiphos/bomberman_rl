from .callbacks import state_to_features, MODEL_NAME, ACTIONS, FEATURE_SIZE
from helper import gameStateSymmetry, actionSym, SYMMETRIES
import events as e
from collections import namedtuple, deque

import os
import pickle
import random
import time
import datetime
from typing import List
import numpy as np
np.seterr(all='raise')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters
TRAIN_EVERY_N_GAMES = 20

TRAIN_BATCH_SIZE = 10000
TRANSITION_HISTORY_SIZE = 500000  # keep only ... last transitions
GAMMA = 0.99


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.round_rewards = []
    self.current_round_reward = 0
    self.game_counter = 0
    self.round = 0
    if os.path.exists("logs/score.txt"):
        os.remove("logs/score.txt")

    self.model_dir = datetime.datetime.now().isoformat(timespec='minutes')
    if not os.path.exists(f"good_models/{self.model_dir}"):
        os.mkdir(f"good_models/{self.model_dir}")


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
        for sym in SYMMETRIES:
            features = state_to_features(gameStateSymmetry(old_game_state, sym), self.logger)
            newFeatures = state_to_features(gameStateSymmetry(new_game_state, sym), self.logger)
            self_action = actionSym(self_action, sym)
            self.transitions.append(Transition(features, self_action, newFeatures, reward))
            break  # It appears that the symmetry approach worsens performance, we only take the original transition


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    reward = reward_from_events(self, events)
    self.current_round_reward += reward
    self.round_rewards.append(self.current_round_reward)
    self.current_round_reward = 0
    if (last_game_state != None):
        for sym in SYMMETRIES:
            features = state_to_features(gameStateSymmetry(last_game_state, sym), self.logger)
            last_action = actionSym(last_action, sym)
            self.transitions.append(Transition(features, last_action, None, reward))
            break  # It appears that the symmetry approach worsens performance, we only take the original transition

    self.round += 1
    self.logger.info('Round Ended')
    if (self.game_counter < TRAIN_EVERY_N_GAMES):
        self.game_counter += 1
        return
    else:
        self.game_counter = 0
    self.logger.info('Start model update')

    Xs = []
    Ys = []

    if (len(self.transitions) < TRAIN_BATCH_SIZE):
        return

    batch = random.sample(self.transitions, TRAIN_BATCH_SIZE)
    current_features = np.stack([transition[0] for transition in batch])
    current_qs_list = self.model.Q(current_features)

    future_features = np.stack([transition[2] if transition[2] is not None else np.zeros((FEATURE_SIZE)) for transition in batch])
    future_qs_list = self.model.Q(future_features)
    for index, (feature, action, next_feature, reward) in enumerate(batch):
        if type(feature) == type(None):
            continue
        elif type(next_feature) == type(None):
            Y = reward
        else:
            Y = reward + GAMMA * np.amax(future_qs_list[index])

        current_Y = current_qs_list[index]
        current_Y[ACTIONS.index(action)] = Y

        Xs.append(feature)
        Ys.append(current_Y)

    self.model.updateModel(Xs, Ys)
    with open("logs/score.txt", "a") as file:
        log_string = ""
        log_string += f" {self.round}"
        log_string += f" {self.model.forest.oob_score_:.3f}"
        log_string += f" {np.mean(self.round_rewards):.1f}"
        log_string += f" {np.std(self.round_rewards):.1f}"
        log_string += f" {np.amin(self.round_rewards):.1f}"
        log_string += f" {np.amax(self.round_rewards):.1f}"
        file.write(log_string + "\n")

    if np.mean(self.round_rewards) > 50:
        with open(f"good_models/{self.model_dir}/round_{self.round}_avg_{np.mean(self.round_rewards):.0f}.pt", "wb") as file:
            pickle.dump(self.model, file)
    self.round_rewards.clear()
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
