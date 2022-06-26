from __future__ import annotations

import inspect
import struct
import hashlib
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection

from slam.map import OrientedLandmarkSettings, Map, Observation
from slam.action_model import ActionModelSettings, action_model
from slam.resampling import ResampleType
from slam.particle import Particle

@dataclass
class FastSLAMSettings:
    """Settings for the FastSLAM algorithm.
    """
    num_particles: int = 100
    action_model_settings: ActionModelSettings = ActionModelSettings()
    landmark_settings: OrientedLandmarkSettings = OrientedLandmarkSettings()
    map_type: type = type(Map)
    resampling_type: ResampleType = ResampleType.LOW_VARIANCE
    r_std: float = 0.08
    phi_std: float = 10*np.pi/180
    psi_std: float = 5*np.pi/180
    r_std_line: float = 0.08
    phi_std_line: float = 10*np.pi/180
    t0: float = 0.0
    tf: float = np.inf
    visualize: bool = False
    trajectory_trail: bool = False
    rng_seed: int = field(default=None, init=False)

    def __hash__(self) -> int:
        return int(self.hash_str(), 16)

    def hash_str(self) -> str:        
        self.action_model_settings.ODOM_ADD_COV.flags.writeable = False
        self.landmark_settings.cov0.flags.writeable = False
        self.landmark_settings.min_cov.flags.writeable = False
        # ast.parse(inspect.getsource(self.resampling_type)).body[0].value.string
        vars_to_hash =[
            bytearray(struct.pack("f", self.num_particles)),
            self.action_model_settings.action_type.name.encode(),
            self.action_model_settings.uncertainty_type.name.encode(),
            self.action_model_settings.ODOM_ADD_MU.data,
            self.action_model_settings.ODOM_ADD_COV.data,
            self.action_model_settings.ODOM_MULT_MU.data,
            self.action_model_settings.ODOM_MULT_COV.data,
            self.landmark_settings.cov0.data,
            self.landmark_settings.min_cov.data,
            self.map_type.__name__.encode(),
            self.resampling_type.__name__.encode(),
            # bytearray(struct.pack("f", self.t0)),
            # bytearray(struct.pack("f", self.tf)),
            bytearray(struct.pack("f", self.r_std)),
            bytearray(struct.pack("f", self.phi_std)),
            bytearray(struct.pack("f", self.psi_std)),
            bytearray(struct.pack("f", self.r_std_line)),
            bytearray(struct.pack("f", self.phi_std_line)),
            bytearray(struct.pack("f", self.rng_seed if self.rng_seed is not None else -1))
        ]
        all_bytes_joined = b''.join(vars_to_hash)
        return hashlib.sha1(all_bytes_joined).hexdigest()

@dataclass
class SLAMResult:
    """Result of a FastSLAM run.
    """
    map: Map
    trajectory: np.ndarray

class FastSLAM:
    def __init__(self, settings: FastSLAMSettings = FastSLAMSettings(), ax: plt.Axes = None) -> None:
        self.settings: FastSLAMSettings = settings
        self.action_model = lambda pose, odometry: action_model(pose, odometry, self.settings.action_model_settings)
        self.particles: list[Particle] = [Particle(default_landmark_settings=settings.landmark_settings) for _ in range(settings.num_particles)]
        self.particle_markers = [None]*settings.num_particles
        self.n_gain = np.diag([settings.r_std, settings.phi_std, settings.psi_std])
        self.n_gain_line = np.diag([settings.r_std_line, settings.phi_std_line])

        # Contains the estimated location of the robot as a list of (time, [x, y, theta])
        self.trajectory_estimate: list[tuple[int, np.ndarray]] = []
        self.actual_trajectory: list[tuple[int, np.ndarray]] = []

        if settings.visualize:
            if ax is None:
                _, ax = plt.subplots()
                self.ax = ax
            else:
                self.ax = ax
            self._init_visualizer()

        self.cur_time: float = 0

        if settings.rng_seed is not None:
            np.random.seed(settings.rng_seed)

    def perform_action(self, t: float, odometry: np.ndarray, actual_location: np.ndarray = None) -> None:
        """Update the pose of all particles using the odometry data.

        Args:
            odometry: The odometry data as a numpy array of [dx, dy, dtheta]
        """

        self.trajectory_estimate += [(self.cur_time, self.pose_estimate())]

        if actual_location is not None:
            self.actual_trajectory += [(self.cur_time, actual_location)]

        def action(pose): return self.action_model(pose, odometry)

        for i in range(len(self.particles)):
            self.particles[i].apply_action(action)        

        if t < self.cur_time:
            print(
                f'[WARNING] ({inspect.currentframe().f_code.co_name}) Time is going backwards!\n\tLatest sample time: {self.cur_time}\n\tNew sample time: {t}')
        self.cur_time = t
    def make_unoriented_observation(self, t: float, obs_data: tuple[int, tuple[float, float]]) -> None:
        """Updates all particles' maps using the observation data, and 
        reweighs the particles based on the likelihood of the observation.

        Args:
            observation: The observation data as a tuple of (id, [r (m), phi (rad), psi(rad)])
        """
        changed_mask = np.empty((len(self.particles),), dtype=bool)
        for i, particle in enumerate(self.particles):
            changed_mask[i] = particle.make_oriented_observation(obs_data, self.n_gain[:2, :2])

        self._normalize_some_particle_weights(changed_mask)

        if t < self.cur_time:
            print(
                f'[WARNING] ({inspect.currentframe().f_code.co_name}) Time is going backwards!\n\tLatest sample time: {self.cur_time}\n\tNew sample time: {t}')

    def make_oriented_observation(self, t: float, obs_data: tuple[int, tuple[float, float, float]]) -> None:
        """Updates all particles' maps using the observation data, and 
        reweighs the particles based on the likelihood of the observation.

        Args:
            observation: The observation data as a tuple of (id, [r (m), phi (rad), psi(rad)])
        """
        changed_mask = np.empty((len(self.particles),), dtype=bool)
        for i, particle in enumerate(self.particles):
            changed_mask[i] = particle.make_oriented_observation(obs_data, self.n_gain)

        self._normalize_some_particle_weights(changed_mask)

        if t < self.cur_time:
            print(
                f'[WARNING] ({inspect.currentframe().f_code.co_name}) Time is going backwards!\n\tLatest sample time: {self.cur_time}\n\tNew sample time: {t}')

    def make_line_observation(self, t: float, obs_data: tuple[int, tuple[float, float]]) -> None:
        """Updates all particles' maps using the observation data, and 
        reweighs the particles based on the likelihood of the observation.

        Args:
            observation: The observation data as a tuple of (id, [rh (m), th (rad)])
                    representing a line in the map.
        """
        changed_mask = np.empty((len(self.particles),), dtype=bool)
        for i, particle in enumerate(self.particles):
            changed_mask[i] = particle.make_line_observation(obs_data, self.n_gain_line)

        self._normalize_some_particle_weights(changed_mask)

        if t < self.cur_time:
            print(
                f'[WARNING] ({inspect.currentframe().f_code.co_name}) Time is going backwards!\n\tLatest sample time: {self.cur_time}\n\tNew sample time: {t}')

    def resample(self) -> None:
        """Resamples the particles based on their weights.
        """
        weights = np.array([particle.weight for particle in self.particles])
        if any(weights - weights[0] != 0):
            self.particles = self.settings.resampling_type(self.particles, weights, self.settings.num_particles)

    def pose_estimate(self) -> np.ndarray:
        """Returns the estimated location of the robot.

        Returns:
            The estimated location of the robot as a numpy array of [x, y, theta]
        """
        return np.sum([particle.pose * particle.weight for particle in self.particles],
                      axis=0) / np.sum([particle.weight for particle in self.particles])

    def map_estimate(self) -> Map:
        """Returns the map of the robot.

        Returns:
            The map of the robot.
        """
        particle_idx_for_map = np.argmax(np.array([particle.weight for particle in self.particles]), axis=0)
        map_estimate: Map = self.particles[particle_idx_for_map].map
        return map_estimate

    def end(self) -> SLAMResult:
        """Returns the final SLAM result.

        Returns:
            The final SLAM result.
        """
        map = self.map_estimate() # prepare map for pickling in order to remove matplotlib-caused huge pickled files
        trajectory_estimate = self.trajectory_estimate
        return SLAMResult(map=map, trajectory=trajectory_estimate)

    def _normalize_particle_weights(self) -> None:
        weights = np.array([particle.weight for particle in self.particles])
        weights = weights / weights.sum()
        for i, particle in enumerate(self.particles):
            particle.weight = weights[i]

    def _normalize_some_particle_weights(self, changed_mask : np.ndarray(dtype=bool)) -> None:  
        """ Keeps not changed_mask particles with same probability, normalizes the rest.
            Assumes that the weights summed to 1 before being changed
        """      
        weights = np.array([particle.weight for particle in self.particles])
        weights[changed_mask] = (1 - weights[~changed_mask].sum()) * weights[changed_mask] / weights[changed_mask].sum()
        for i, particle in enumerate(self.particles):
            particle.weight = weights[i]

    # Visualization methods (if settings.visualize is True)
    def _init_visualizer(self, ylim: tuple = (-3, 3), xlim: tuple = (-3, 3)) -> None:
        self.actual_location_dot: PathCollection = self.ax.scatter(0, 0, marker='x', c='k', alpha=0.0)
        self.drawn_map_estimate = self.particles[0].map  # guess map just to avoid initializing to None
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)

        for idx, particle in enumerate(self.particles):
            self.particle_markers[idx] = self.ax.plot(0, 0, c='C00')[0]

        self.ax.set_autoscale_on(True)
        self.ax.axes.set_aspect('equal')

        x = 6
        self.ax.plot([-x, x, x, -x], [-x, -x, x, x], c='k', linewidth=0)

        if self.settings.trajectory_trail:
            self.actual_trajectory_trail, = self.ax.plot([], [], c='C03', linewidth=0.5, label='Actual Trajectory')
            self.estimated_trajectory_trail, = self.ax.plot(
                [], [], c='C04', linewidth=0.5, label='Estimated trajectory')

    def _draw_location(self) -> None:
        for idx, particle in enumerate(self.particles):
            particle._draw(self.particle_markers[idx])
        if self.actual_trajectory:
            self.actual_location_dot.set(offsets = [self.actual_trajectory[-1][1][:2]])

        if self.settings.trajectory_trail:
            pose_estimate = self.pose_estimate()
            if self.actual_trajectory:
                actual_traj_posx = [tupl[1][0] for tupl in self.actual_trajectory]
                actual_traj_posy = [tupl[1][1] for tupl in self.actual_trajectory]
                self.actual_trajectory_trail.set_data(actual_traj_posx, actual_traj_posy)
            prev_traj_est = self.estimated_trajectory_trail.get_data(orig=True)
            self.estimated_trajectory_trail.set_data(
                list(prev_traj_est[0]) + [pose_estimate[0]],
                list(prev_traj_est[1]) + [pose_estimate[1]])

    def _draw_map(self) -> None:
        particle_idx_for_map = np.argmax(np.array([particle.weight for particle in self.particles]), axis=0)
        new = self.particles[particle_idx_for_map].map
        if self.drawn_map_estimate is not new:
            self.drawn_map_estimate._undraw()
            self.drawn_map_estimate = new

        self.drawn_map_estimate._draw(self.ax, color_ellipse='C01', color_p='C01', color_z='C01')

    def _draw(self) -> None:
        # self.ax.relim()
        self._draw_location()
        self._draw_map()
        self.ax.autoscale_view(False, True, True)
        plt.pause(0.01)
        plt.show(block=False)

if __name__ == '__main__':
    from math import *
    import time
    slam_settings = FastSLAMSettings(
        num_particles=1,
        action_model_settings=ActionModelSettings(
        ),
        landmark_settings=OrientedLandmarkSettings(
        ),
        resampling_type=ResampleType.UNIFORM,
        visualize=True
    )

    slam = FastSLAM(slam_settings)
    t0 = time.time()
    def loc(location, movement):
        return action_model(location, movement, ActionModelSettings(
            POSE_ADD_COV=np.diag([0.0, 0.0, 0.0]),
            ODOM_ADD_COV=np.diag([0.0, 0.0]),
            ))
    cur_loc = np.array([0, 0, 0])
    def act(movement):
        movement = np.array(movement)
        global cur_loc
        cur_loc = loc(cur_loc, movement)
        slam.perform_action(time.time()-t0, movement, cur_loc)

    def observe(id, position):
        slam.make_unoriented_observation(time.time()-t0, (id, position))
    act([0.0, 0.0])
    observe(0, [0.0, 0.0])
