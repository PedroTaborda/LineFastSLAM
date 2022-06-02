from __future__ import annotations

from typing import Callable
import copy

import numpy as np
import matplotlib.pyplot as plt

from slam.map import Map, Observation


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

    def __init__(self, map: Map = None, pose=(0, 0, 0), weight: float = 1.0) -> None:
        if map is None:
            # Brand new particle being created: prepare new everything
            self.map = Map()
            self.pose = np.array(pose)  # np.random.uniform(low=-1, high=1, size=3)
            self.weight = 1.0
            return
        self.map: Map = map
        self.pose: np.ndarray = np.array(pose)
        self.weight: float = weight

    def apply_action(self, action: Callable[[np.ndarray], np.ndarray]) -> None:
        self.pose = action(self.pose)

    def make_observation(self, obs_data: tuple[int, tuple[float, float]], n_gain: np.ndarray) -> None:
        """Make an observation of a landmark on the map.

        Side effects:
            -The map is updated with the observation (a landmark may be added)
            -The particle's weight is updated
        """
        px, py, theta = self.pose
        
        r, phi = obs_data[1]
        p = np.array([px, py])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        def h(x, n):    # z is observed position of landmark in robot's reference frame
            z_no_noise = R @ (x - p)
            r_err, ang_err = n_gain @ n
            R_error = np.array([[np.cos(ang_err), np.sin(ang_err)], [-np.sin(ang_err), np.cos(ang_err)]])
            return (1 + (r_err/np.linalg.norm(z_no_noise))) * R_error @ z_no_noise

        def h_inv(z):
            return R.T @ z + p

        def get_Dhx(x):
            return R

        def get_Dhn(x):
            z = R @ (x - p)
            return np.array([[z[0], -z[1]], [z[1], z[0]]]) @ n_gain
        
        obs = Observation(
            landmark_id=obs_data[0],
            z=np.array([r*np.cos(phi), r*np.sin(phi)]),
            h=h,
            h_inv=h_inv,
            get_Dhx=get_Dhx,
            get_Dhn=get_Dhn,
        )
        self.weight *= self.map.update(obs)

    def copy(self) -> Particle:
        """Copy the particle, creating a new particle sharing the same map.
        """
        return Particle(self.map.copy(), copy.copy(self.pose), copy.copy(self.weight))

    def _draw(self, line: plt.Line2D) -> None:
        R = np.array([[np.cos(np.deg2rad(self.pose[2])), -np.sin(np.deg2rad(self.pose[2]))],
                      [np.sin(np.deg2rad(self.pose[2])), np.cos(np.deg2rad(self.pose[2]))]])
        arrow = (R @ self.canonical_arrow.T)
        arrow = (arrow.T + self.pose[:2]).T
        line.set_data(arrow[0, :], arrow[1, :])

    def __repr__(self) -> str:
        return f'Particle(pose={self.pose}, weight={self.weight})'

    def __str__(self) -> str:
        return self.__repr__()
