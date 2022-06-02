from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from slam.map import Landmark, LandmarkSettings, Map, Observation
from slam.action_model import ActionModelSettings, action_model
from slam.resampling import ResampleType
from slam.particle import Particle

@dataclass
class FastSLAMSettings:
    """Settings for the FastSLAM algorithm.
    """
    num_particles: int = 100
    action_model_settings: ActionModelSettings = ActionModelSettings()
    landmark_settings: LandmarkSettings = LandmarkSettings()
    resampling_type: ResampleType = ResampleType.UNIFORM
    r_std: float = 0.1
    phi_std: float = 0.1
    visualize: bool = False

class FastSLAM:
    def __init__(self, settings: FastSLAMSettings = FastSLAMSettings()) -> None:
        self.settings: FastSLAMSettings = settings
        self.action_model = lambda pose, odometry: action_model(pose, odometry, self.settings.action_model_settings)
        self.particles: list[Particle] = [Particle() for _ in range(settings.num_particles)]
        self.particle_markers = [None]*settings.num_particles
        self.n_gain = np.diag([settings.r_std, settings.phi_std])

        if settings.visualize:
            self._init_visualizer()

    def perform_action(self, odometry: np.ndarray, actual_location: np.ndarray = None) -> None:
        """Update the pose of all particles using the odometry data.

        Args:
            odometry: The odometry data as a numpy array of [dx, dy, dtheta]
        """
        old_particles = copy.copy(self.particles)
        action = lambda pose: self.action_model(pose, odometry)
        self._normalize_particle_weights()
        weights = np.array([particle.weight for particle in self.particles])
        for i, picked in enumerate(np.random.choice(len(self.particles), self.settings.num_particles, replace=True, p=weights)):
            self.particles[i] = old_particles[picked].copy()
            self.particles[i].apply_action(action)
            self.particles[i].weight = 1/self.settings.num_particles

        if self.settings.visualize:
            self._draw_location(actual_location=actual_location)
            
    def make_observation(self, obs_data: tuple[int, tuple[float, float]]) -> None:
        """Updates all particles' maps using the observation data, and 
        reweighs the particles based on the likelihood of the observation.

        Args:
            observation: The observation data as a tuple of (id, [r (m), phi (rad)])
        """
        for particle in self.particles:
            particle.make_observation(obs_data, self.n_gain)
        
        weights = np.array([particle.weight for particle in self.particles])
        print(weights)
        self._normalize_particle_weights()
        weights = np.array([particle.weight for particle in self.particles])
        print(weights)

        if self.settings.visualize:
            self._draw_map()
        a = 1 # for debug purposes

    def resample(self) -> None:
        """Resamples the particles based on their weights.
        """
        weights = np.array([particle.weight for particle in self.particles])
        self.particles = self.settings.resampling_type(self.particles, weights)

    def get_location(self) -> np.ndarray:
        """Returns the estimated location of the robot.

        Returns:
            The estimated location of the robot as a numpy array of [x, y, theta]
        """
        return np.array([particle.pose for particle in self.particles]).mean(axis=0)

    def get_map(self) -> Map:
        """Returns the map of the robot.

        Returns:
            The map of the robot.
        """
        return self.particles[0].map
    
    def _normalize_particle_weights(self) -> None:
        weights = np.array([particle.weight for particle in self.particles])
        weights = weights / weights.sum()
        for i, particle in enumerate(self.particles):
            particle.weight = weights[i]

    # Visualization methods (if settings.visualize is True)
    def _init_visualizer(self, ylim: tuple=(-3, 3), xlim: tuple=(-3, 3)) -> None:
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        self.actual_location_dot: PathCollection = self.ax.scatter(0, 0, marker='x', c='k', alpha=0.0)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

        for idx, particle in enumerate(self.particles):
            self.particle_markers[idx] = self.ax.plot(0, 0, c='C00')[0]
        
        self.ax.set_autoscale_on(True)
        self.ax.axes.set_aspect('equal')

        x = 6
        self.ax.plot([-x, x, x, -x], [-x, -x, x, x], c='k', linewidth=0)
    
    def _draw_location(self, actual_location: np.ndarray = None) -> None:
        for idx, particle in enumerate(self.particles):
            particle._draw(self.particle_markers[idx])
        if actual_location is not None:
            self.actual_location_dot.set(offsets = [actual_location[:2]])

        self._draw()

    def _draw_map(self, actual_map = None) -> None:
        particle_idx_for_map = np.argmax(np.array([particle.weight for particle in self.particles]), axis=0)
        map_estimate: Map = self.particles[particle_idx_for_map].map
        map_estimate._draw(self.ax, color_ellipse='C01', color_p='C01', color_z='C01')
        self._draw()

    def _draw(self) -> None:
        self.ax.relim()
        self.ax.autoscale_view(False,True,True)
        plt.pause(0.01)
        plt.show(block=False)

if __name__ == '__main__':
    from math import *
    slam_settings = FastSLAMSettings(
        num_particles=1,
        action_model_settings=ActionModelSettings(
        ),
        landmark_settings=LandmarkSettings(
        ),
        resampling_type=ResampleType.UNIFORM,
        visualize=True
    )

    slam = FastSLAM(slam_settings)
    def loc(location, movement):
        return action_model(location, movement, ActionModelSettings(
            uncertainty_additive_covariance=np.diag([0.0, 0.0, 0.0]),
            uncertainty_multiplicative_covariance=np.diag([0.0, 0.0]),
            ))
    cur_loc = np.array([0, 0, 0])
    def act(movement):
        movement = np.array(movement)
        global cur_loc
        cur_loc = loc(cur_loc, movement)
        slam.perform_action(movement, cur_loc)

    def observe(id, position):
        slam.make_observation(Observation(landmark_id=id, z=position))
    act([0.0, 0.0])
    observe(0, [0.0, 0.0])