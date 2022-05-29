from __future__ import annotations

from typing import Callable
import copy

import numpy as np
import matplotlib.pyplot as plt

from slam.map import Map


class Particle:
    canonical_arrow: np.ndarray = np.array(
        [[0, 0],
        [1, 0],
        [0.5, 0.5],
        [1, 0],
        [0.5, -0.5],
        [1, 0],
        ])*0.1
    arrow_size = 0.1
    def __init__(self, map: Map = None, pose = (0, 0, 0), weight: float = 1.0) -> None:
        if map is None:
            # Brand new particle being created: prepare new everything
            self.map = Map()
            self.pose = np.random.uniform(low=-1, high=1, size=3)
            self.weight = 1.0
            return
        self.map: Map = map
        self.pose: np.ndarray = np.array(pose)
        self.weight: float = weight


    def apply_action(self, action: Callable[[np.ndarray], np.ndarray]) -> None:
        self.pose = action(self.pose)

    def make_observation(self, observation: tuple[int, np.ndarray]) -> None:
        """Make an observation of a landmark on the map.

        Args:
            observation: A tuple of the form (landmark_id, landmark_position)

        Side effects:
            -The map is updated with the observation (a landmark may be added)
            -The particle's weight is updated
        """
        self.weight *= self.map.update(self.pose, observation)

    def copy(self) -> Particle:
        """Copy the particle, creating a new particle sharing the same map.
        """
        return Particle(self.map, copy.copy(self.pose), copy.copy(self.weight))

    def _draw(self, line: plt.Line2D, **plot_args) -> None:
        R = np.array([[np.cos(self.pose[2]), -np.sin(self.pose[2])],
                        [np.sin(self.pose[2]), np.cos(self.pose[2])]])
        arrow = (R @ self.canonical_arrow.T)
        arrow = (arrow.T + self.pose[:2]).T
        line.set_data(arrow[0, :], arrow[1, :])

    def __repr__(self) -> str:
        return f'Particle(pose={self.pose}, weight={self.weight})'
    
    def __str__(self) -> str:
        return self.__repr__()
