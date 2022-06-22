from __future__ import annotations

from typing import Callable
import copy

import numpy as np
import matplotlib.pyplot as plt

from slam.map import OrientedLandmarkSettings, Map, Observation, UnorientedObservation, LineObservation, get_Dhn_line, get_Dhx_line, h_inv_line, h_line

import scipy.stats
import time

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

    def __init__(self, map: Map = None, pose=(0, 0, 0), weight: float = 1.0, default_landmark_settings=None) -> None:
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

    def make_line_observation(self, obs_data: tuple[int, tuple[float, float]], n_gain: np.ndarray) -> None:
        """Observe a line on the map. Measurements are in the robot's reference frame.

        Args:
            obs_data: A tuple of the form (landmark_id, (rh, th)),
            where rh, th describes a line in the robot's reference frame as 
            (orthogonal distance, angle from robot's heading).

        Side effects:
            -The map is updated with the observation (a landmark may be added)
            -The particle's weight is updated
        """

        px, py, theta = self.pose

        rh, th = obs_data[1]
        p = np.array([px, py])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        lidar_vector = np.array([-0.0625, 0])
        parameters = (p, theta, R, lidar_vector, n_gain)

        def diff(rh_th1, rh_th2):
            return np.array([rh_th1[0] - rh_th2[0], np.mod(rh_th1[1] - rh_th2[1] + np.pi, 2*np.pi) - np.pi])

        observed_landmarks = self.map.landmarks
        observed_lines_keys = [landmark for landmark in observed_landmarks if landmark < 0]      
        max_likelihood, best_key = 0, 0
        for key in observed_lines_keys:
            likelihood = self.map.landmarks[key].get_likelihood(np.array([rh, th]), diff = diff, parameters = parameters)
            if likelihood > max_likelihood:
                max_likelihood, best_key = likelihood, key
        landmark_id = best_key

        if max_likelihood < 0.2:
            landmark_id = min(observed_lines_keys) - 1 if observed_lines_keys else -1

        obs = LineObservation(
            landmark_id=landmark_id,
            z=np.array([rh, th]),
            h=h_line,
            h_inv=h_inv_line,
            get_Dhx=get_Dhx_line,
            get_Dhn=get_Dhn_line
        )
        self.weight *= self.map.update(obs, diff = diff, parameters = parameters)
        
    def make_unoriented_observation(self, obs_data: tuple[int, tuple[float, float]], n_gain: np.ndarray) -> None:
        """Make an observation of a landmark on the map.

        Side effects:
            -The map is updated with the observation (a landmark may be added)
            -The particle's weight is updated
        """
        px, py, theta = self.pose
        
        r, phi = obs_data[1]
        p = np.array([px, py])
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

        def h(x, n, parameters):    # z is observed position of landmark in robot's reference frame
            z_no_noise = R @ (x - p)
            r_err, ang_err = n_gain @ n
            R_error = np.array([[np.cos(ang_err), np.sin(ang_err)], [-np.sin(ang_err), np.cos(ang_err)]])
            return (1 + (r_err/np.linalg.norm(z_no_noise))) * R_error @ z_no_noise

        def h_inv(z, parameters):
            return R.T @ z + p

        def get_Dhx(x, parameters):
            return R

        def get_Dhn(x, parameters):
            z = R @ (x - p)
            return np.array([[z[0], -z[1]], [z[1], z[0]]]) @ n_gain
        
        obs = UnorientedObservation(
            landmark_id=obs_data[0]+100,
            z=np.array([r*np.cos(phi), r*np.sin(phi)]),
            h=h,
            h_inv=h_inv,
            get_Dhx=get_Dhx,
            get_Dhn=get_Dhn,
        )
        self.weight *= self.map.update(obs)

    def make_oriented_observation(self, obs_data: tuple[int, tuple[float, float, float]], n_gain: np.ndarray) -> None:
        """Make an observation of a landmark on the map, considering that the landmark is an Aruco.

        Side effects:
            -The map is updated with the observation (a landmark may be added)
            -The particle's weight is updated
        """
        px, py, theta = self.pose
        
        r, phi, psi = obs_data[1]
        p = np.array([px, py])
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

        def h(x, n, parameters):    # z is observed position of landmark in robot's reference frame
            x_prime = x[0:2]
            psi_prime = x[2]
            z_no_noise = R @ (x_prime - p)
            z_psi_no_noise = psi_prime - theta
            r_err, ang_err, psi_err = n_gain @ n
            R_error = np.array([[np.cos(ang_err), np.sin(ang_err)], [-np.sin(ang_err), np.cos(ang_err)]])
            return np.array([*((1 + (r_err/np.linalg.norm(z_no_noise))) * R_error @ z_no_noise), z_psi_no_noise + psi_err])

        def h_inv(z, parameters):
            z_prime = z[0:2]
            z_psi = z[2]
            return np.array([*(R.T @ z_prime + p), z_psi + theta])

        def get_Dhx(x, parameters):
            Dh = np.zeros((3,3))
            Dh[0:2, 0:2] = R
            Dh[2, 2] = 1
            return Dh

        def get_Dhn(x, parameters):
            Dh = np.zeros((3,3))
            z = R @ (x[0:2] - p)
            Dhz = np.array([[z[0], -z[1]], [z[1], z[0]]]) @ n_gain[0:2, 0:2]
            Dh[0:2, 0:2] = Dhz
            Dh[2, 2] = n_gain[2, 2]
            return Dh
        
        obs = Observation(
            landmark_id=obs_data[0],
            z=np.array([r*np.cos(phi), r*np.sin(phi), psi]),
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
        R = np.array([[np.cos(self.pose[2]), -np.sin(self.pose[2])],
                      [np.sin(self.pose[2]), np.cos(self.pose[2])]])
        arrow = (R @ self.canonical_arrow.T)
        arrow = (arrow.T + self.pose[:2]).T
        line.set_data(arrow[0, :], arrow[1, :])

    def __repr__(self) -> str:
        return f'Particle(pose={self.pose}, weight={self.weight})'

    def __str__(self) -> str:
        return self.__repr__()
