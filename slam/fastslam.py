from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from slam.map import Map
from slam.action_model import ActionModelSettings, action_model
from slam.ekf import EKFSettings
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

        if settings.visualize:
            self._init_visualizer()

    def perform_action(self, odometry: np.ndarray) -> None:
        """Update the pose of all particles using the odometry data.

        Args:
            odometry: The odometry data as a numpy array of [dx, dy, dtheta]
        """
        old_particles = self.particles
        action = lambda pose: self.action_model(pose, odometry)
        for i, picked in enumerate(np.random.choice(len(self.particles), self.settings.num_particles, replace=True)):
            self.particles[i] = old_particles[picked].copy()
            self.particles[i].apply_action(action)
            
    def make_observation(self, observation: tuple[int, np.ndarray]) -> None:
        """Updates all particles' maps using the observation data, and 
        reweighs the particles based on the likelihood of the observation.

        Args:
            observation: The observation data as a tuple of (id, [x, y(, theta)])
        """
        for particle in self.particles:
            particle.make_observation(observation)

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
    def _init_visualizer(self) -> None:
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 5))
        self.particle_dots: PathCollection = self.ax.scatter(np.zeros((self.settings.num_particles)), np.zeros((self.settings.num_particles)), 'ro')
        self.actual_location_dot: PathCollection = self.ax.scatter(0, 0, 'bo')

    def _draw_location(self, actual_location: np.ndarray = None) -> None:
        self.particle_dots.set_paths(np.array([particle.pose[:2] for particle in self.particles]))

    def _draw_map(self, actual_map = None) -> None:
        ...


if __name__ == '__main__':
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
        resampling_type=ResampleType.UNIFORM
    )

    slam = FastSLAM(slam_settings)
