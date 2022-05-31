from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from slam.map import Map
from slam.action_model import ActionModelSettings, action_model
from ekf.ekf import EKFSettings
from slam.resampling import ResampleType
from slam.particle import Particle

@dataclass
class FastSLAMSettings:
    """Settings for the FastSLAM algorithm.
    """
    num_particles: int = 100
    action_model_settings: ActionModelSettings = ActionModelSettings()
    ekf_settings: EKFSettings = EKFSettings()
    resampling_type: ResampleType = ResampleType.UNIFORM
    visualize: bool = False

class FastSLAM:
    def __init__(self, settings: FastSLAMSettings = FastSLAMSettings()) -> None:
        self.settings: FastSLAMSettings = settings
        self.action_model = lambda pose, odometry: action_model(pose, odometry, self.settings.action_model_settings)
        self.particles: list[Particle] = [Particle() for _ in range(settings.num_particles)]
        self.particle_markers = [None]*settings.num_particles

        if settings.visualize:
            self._init_visualizer()

    def perform_action(self, odometry: np.ndarray, actual_location: np.ndarray = None) -> None:
        """Update the pose of all particles using the odometry data.

        Args:
            odometry: The odometry data as a numpy array of [dx, dy, dtheta]
        """
        old_particles = self.particles
        action = lambda pose: self.action_model(pose, odometry)
        for i, picked in enumerate(np.random.choice(len(self.particles), self.settings.num_particles, replace=True)):
            self.particles[i] = old_particles[picked].copy()
            self.particles[i].apply_action(action)

        if self.settings.visualize:
            self._draw_location(actual_location=actual_location)
            
    def make_observation(self, observation: tuple[int, np.ndarray]) -> None:
        """Updates all particles' maps using the observation data, and 
        reweighs the particles based on the likelihood of the observation.

        Args:
            observation: The observation data as a tuple of (id, [x, y(, theta)])
        """
        for particle in self.particles:
            particle.make_observation(observation)

        if self.settings.visualize:
            self._draw_map()

    def resample(self) -> None:
        """Resamples the particles based on their weights.
        """
        weights = np.array([particle.weight for particle in self.particles])
        self.particles = self.settings.resampling_type.value(weights)

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
    
    # Visualization methods (if settings.visualize is True)
    def _init_visualizer(self, ylim: tuple=(-3, 3), xlim: tuple=(-3, 3)) -> None:
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        self.actual_location_dot: PathCollection = self.ax.scatter(0, 0, marker='x', c='C01')
        self.landmark_dots: PathCollection = self.ax.scatter(0, 0, marker='^', c='C00')
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

        for idx, particle in enumerate(self.particles):
            self.particle_markers[idx] = self.ax.plot(0, 0, c='C01')[0]

    
    def _draw_location(self, actual_location: np.ndarray = None) -> None:
        for idx, particle in enumerate(self.particles):
            particle._draw(self.particle_markers[idx], color='C01')
        if actual_location is not None:
            self.actual_location_dot.set(offsets = [actual_location[:2]])

        self._draw()

    def _draw_map(self, actual_map = None) -> None:
        particle_idx_for_map = np.argmax(np.array([particle.weight for particle in self.particles]), axis=0)
        map_estimate = self.particles[particle_idx_for_map].map.landmarks
        landmark_positions = [map_estimate[landmark].get_mu() for landmark in map_estimate]
        print(landmark_positions)
        self.landmark_dots.set(offsets = landmark_positions)
        self._draw()

    def _draw(self) -> None:
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        plt.show(block=False)

if __name__ == '__main__':
    from math import *
    slam_settings = FastSLAMSettings(
        num_particles=10,
        action_model_settings=ActionModelSettings(
        ),
        ekf_settings=EKFSettings(
            # example
            #initial_covariance=np.diag([0.1, 0.1]),
            #process_noise_covariance=np.diag([0.1, 0.1]),
            #observation_noise_covariance=np.diag([0.1, 0.1]),
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
        slam.make_observation((id, np.array(position)))
    act([0.0, 0.0])
    observe(0, [0.0, 0.0])