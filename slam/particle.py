from __future__ import annotations

from typing import Callable
import copy

import numpy as np
import matplotlib.pyplot as plt

from slam.map import OrientedLandmarkSettings, Map, Observation, UnorientedObservation, LineObservation


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
        pr = R.T @ p
        
        def h_inv(z):
            rh_robot, th_robot = z
            th_world = np.mod(th_robot + theta, 2*np.pi) - np.pi
            point_on_line_world = R @ np.array([rh_robot * np.cos(th_robot), rh_robot * np.sin(th_robot)]) + p
            rh_world = point_on_line_world.dot(np.array([np.cos(th_world), np.sin(th_world)]))
            x = rh_world, th_world
            if rh_world < 0:
                x = -rh_world, np.mod(th_world + np.pi, 2*np.pi) - np.pi
            return np.array(x)

        observed_landmarks = self.map.landmarks
        observed_lines_keys = [landmark for landmark in observed_landmarks if landmark < 0]
        rh_world, th_world = h_inv(np.array([rh, th]))
        xmark = np.array([rh_world*np.cos(th_world), rh_world*np.sin(th_world)])
        try_to_match = []
        for key in observed_lines_keys:
            rh_observed_world, th_observed_world = self.map.landmarks[key].get_mu()
            xmark_observed = np.array([rh_observed_world*np.cos(th_observed_world), rh_observed_world*np.sin(th_observed_world)])
            if np.linalg.norm(xmark_observed - xmark) < 1: # TODO: make this a parameter (it means only try matching close landmarks)
                try_to_match.append(key)

        if try_to_match:
            max_likelihood, best_key = 0, 0
            for key in try_to_match:
                likelihood = self.map.landmarks[key].get_likelihood(np.array([rh, th]), diff = lambda t1, t2 : np.mod(t1 - t2, 2*np.pi) - np.pi)
                if likelihood > max_likelihood:
                    max_likelihood, best_key = likelihood, key
            landmark_id = best_key
        else:
            landmark_id = min(observed_lines_keys) - 1 if observed_lines_keys else -1


        def h(x, n):
            rh_world, th_world = x
            th_robot = np.mod(th_world - theta, 2*np.pi) - np.pi
            point_on_line_robot = R.T @ (np.array([rh_world * np.cos(th_world), rh_world * np.sin(th_world)]) - p)
            rh_robot = point_on_line_robot.dot(np.array([np.cos(th_robot), np.sin(th_robot)]))
            z = [rh_robot, th_robot]
            if rh_robot < 0:
                z = [-rh_robot, np.mod(th_robot + np.pi, 2*np.pi) - np.pi]
            return np.array(z) + n_gain @ n


        def get_Dhx(x):
            dhx = np.eye(2)
            dhx[0, 0] = np.cos(x[1] - theta)
            dhx[0, 1] = pr[1] * np.cos(x[1] - theta) - (x[0] + pr[0]) * np.sin(x[1] - theta)
            return dhx

        def get_Dhn(x):
            return n_gain

        obs = LineObservation(
            landmark_id=landmark_id,
            z=np.array([rh, th]),
            h=h,
            h_inv=h_inv,
            get_Dhx=get_Dhx,
            get_Dhn=get_Dhn
        )
        self.weight *= self.map.update(obs, diff = lambda t1, t2 : np.mod(t1 - t2, 2*np.pi) - np.pi)
        
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

        def h(x, n):    # z is observed position of landmark in robot's reference frame
            x_prime = x[0:2]
            psi_prime = x[2]
            z_no_noise = R @ (x_prime - p)
            z_psi_no_noise = psi_prime - theta
            r_err, ang_err, psi_err = n_gain @ n
            R_error = np.array([[np.cos(ang_err), np.sin(ang_err)], [-np.sin(ang_err), np.cos(ang_err)]])
            return np.array([*((1 + (r_err/np.linalg.norm(z_no_noise))) * R_error @ z_no_noise), z_psi_no_noise + psi_err])

        def h_inv(z):
            z_prime = z[0:2]
            z_psi = z[2]
            return np.array([*(R.T @ z_prime + p), z_psi + theta])

        def get_Dhx(x):
            Dh = np.zeros((3,3))
            Dh[0:2, 0:2] = R
            Dh[2, 2] = 1
            return Dh

        def get_Dhn(x):
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
