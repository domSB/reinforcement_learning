"""
Eigene Implementierung von Minesweeper.
"""
import numpy as np
from scipy.signal import convolve2d


def create_field(size, difficulty):
    arr = np.zeros((size, size))
    mines_count = int(difficulty * size ** 2)
    mines_count = max(mines_count, 2)
    idx = tuple(np.random.choice(size, (2, mines_count)))
    arr[idx] = 1
    kernel = np.ones((3, 3))
    field = convolve2d(arr, kernel, mode='same')
    mask = arr.astype(bool)
    field[mask] = -9
    field = field / 9
    return field


def create_action_probs_from_tuple(action):
    """
    Hilfsfunktion, damit Menschen das Spiel mit einer deterministischen
    Strategie spielen können
    """
    global size
    probs = np.zeros((size, size))
    probs[action] = 1
    return probs


class Environment:

    def __init__(self, size, difficulty):
        self.true_field = create_field(size, difficulty)
        self.visible_field = np.ones((size, size)) * -1
        self.flattened_actions = []
        for i in range(size):
            for j in range(size):
                self.flattened_actions.append((i, j))
        self.action_num = len(self.flattened_actions)
        self.done = False
        self.size, self.diff = size, difficulty
        self.free_places = (self.true_field != -1).sum()
        self.free_places_to_go = self.free_places

    def reset(self):
        self.true_field = create_field(self.size, self.diff)
        self.visible_field = np.ones((self.size, self.size)) * -1
        low_val = self.true_field[self.true_field >= 0].min()
        start_possibilities = np.argwhere(low_val == self.true_field)
        start_idx = np.random.choice(len(start_possibilities), 1)
        start_position = tuple(start_possibilities[start_idx][0])
        self.visible_field[start_position] = self.true_field[start_position]
        self.done = False
        self.free_places = (self.true_field != -1).sum()
        self.free_places_to_go = self.free_places
        return self.visible_field, 0, self.done

    def make_action(self, action_probs):
        assert not self.done, 'Spiel schon beendet'
        # Stochastische Aktionsauswahl geht bei Numpy nur mir 1-dim arrays
        # Daher der Umweg über geglättete Actionen
        flat_probs = action_probs.reshape(-1)
        action_idx = np.random.choice(self.action_num, 1, p=flat_probs)[0]
        action = self.flattened_actions[action_idx]
        field_value = self.true_field[action]
        if field_value == -1:
            # Auf eine Mine geklickt
            self.done = True
            reward = -1
        else:
            if self.visible_field[action] == -1:
                # Es wurde auf ein verdecktes Feld geklickt
                self.free_places_to_go -= 1
                # Summierte Belohnung bei Gewinn genau 1
                reward = 1 / self.free_places
            else:
                # Es wurde auf ein offenes Feld geklickt
                reward = 0
            self.visible_field[action] = field_value
            self.done = self.free_places_to_go == 0

        return self.visible_field, reward, self.done

