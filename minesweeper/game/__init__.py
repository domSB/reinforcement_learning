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


class Environment:

    def __init__(self, size, difficulty):
        self.field = create_field(size, difficulty)
