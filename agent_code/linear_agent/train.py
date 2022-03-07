from .callbacks import state_to_features, MODEL_NAME, ACTIONS
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
TRANSITION_HISTORY_SIZE = 30  # keep only ... last transitions
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
    self.transitions.append(Transition(state_to_features(old_game_state, self.logger), self_action, state_to_features(new_game_state, self.logger), reward_from_events(self, events)))
    feat=state_to_features(old_game_state, self.logger)
    feat_new=state_to_features(new_game_state, self.logger)
    if type(feat) == type(None) or type(feat_new) == type(None):
                pass
    else:
        mirx_feat, mirx_action = mirrorx(feat,self_action)
        mirx_feat_new, _ = mirrorx(feat_new, self_action)
        miry_feat, miry_action = mirrory(feat,self_action)
        miry_feat_new, _ = mirrory(feat_new, self_action)
        self.transitions.append(Transition(mirx_feat, mirx_action, mirx_feat_new, reward_from_events(self, events)))
        self.transitions.append(Transition(miry_feat, miry_action, miry_feat_new, reward_from_events(self, events)))

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state, self.logger), last_action, state_to_features(last_game_state, self.logger), reward_from_events(self, events)))

    self.logger.info('Round Ended. Start model update')
    oldBeta = self.model.beta
    for i, _ in enumerate(ACTIONS):
        summe = 0
        for (feature, action, next_feature, reward) in self.transitions:
            if type(feature) == type(None) or type(next_feature) == type(None):
                continue
            Qs = [self.model.Q(next_feature, a) for a in range(len(ACTIONS))]
            Y = reward + GAMMA * np.max(Qs)
            summe += np.transpose(feature) * (Y - feature @ oldBeta[i])

        newBeta = oldBeta[i] + ALPHA/len(self.transitions) * summe
        self.logger.debug(f'Updated beta for action {ACTIONS[i]}')
        self.model.updateBeta(i, newBeta)

    self.transitions.clear()
    self.model.EXPLORATION_RATE = max(0.05, self.model.EXPLORATION_RATE*0.96)

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
        #e.COIN_COLLECTED: 2,
        #e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -0.5,
        #e.KILLED_SELF: -5,
        e.MOVED_UP: 0.5,   #moved:0.5 waited: -0.2
        e.MOVED_DOWN: 0.5,
        e.MOVED_RIGHT: 0.5,
        e.MOVED_LEFT: 0.5,
        #e.WAITED: -0.2
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def mirrory(features, action):
    if action == 'UP':
        mir_act = 'DOWN'
    elif action == 'DOWN':
        mir_act = 'UP'
    else:
        mir_act = action
    miry_feats= features
   # miry_feats[1] = 1-features[1]
    miry_feats[0] = features[1]
    miry_feats[1] = features[0]
   # miry_feats[7] = 1-features[7]
    #miry_feats[9] = 1-features[9]
    #miry_feats[11] = 1-features[11]
    #miry_feats[13] = -1*features[13]
    return miry_feats, mir_act

def mirrorx(features, action):
    if action == 'LEFT':
        mir_act = 'RIGHT'
    elif action == 'RIGHT':
        mir_act = 'LEFT'
    else:
        mir_act = action
    mirx_feats= features
    #mirx_feats[0] = 1-features[0]
    mirx_feats[2] = features[3]
    mirx_feats[3] = features[2]
    #mirx_feats[6] = 1-features[6]
    #mirx_feats[8] = 1-features[8]
    #mirx_feats[10] = 1-features[10]
    #mirx_feats[12] = -1*features[12]
    return mirx_feats, mir_act