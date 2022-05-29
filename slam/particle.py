from __future__ import annotations

from typing import Callable
import copy

import numpy as np

from slam.map import Map


class Particle:
    def __init__(self, map: Map = None, pose = (0, 0, 0), weight: float = 1.0) -> None:
        if map is None:
            # Brand new particle being created: prepare new everything
            raise NotImplementedError
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

    def __repr__(self) -> str:
        return f'Particle(pose={self.pose}, weight={self.weight})'
    
    def __str__(self) -> str:
        return self.__repr__()
