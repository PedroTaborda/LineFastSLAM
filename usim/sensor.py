from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import math_extra as m

from .robot import Robot
from .umap import UsimMap

@dataclass
class SensorSettings:
    # Sensor parameters
    camera_fov: float = 62.2                    # Field of view in degrees
    camera_range: float = 2                    # Maximum range that the camera can detect the Aruco Markers
    lidar_range: float = 3.5                    # Range in meters
    lidar_angular_resolution: float = 360       # Number of points in one full sweep

    # Sensor noise characterization
    lidar_range_noise_sigma: float = 0          # Standard Deviation of Lidar Range Measurements
    lidar_angular_noise_sigma: float = 0        # Standard Deviation of Lidar Angular Sweep Position
    odometry_angular_noise_sigma: float = 0.1   # Standard Deviation of Odometry Angular Displacement
    odometry_r_noise_sigma: float = 0.1         # Standard Deviation of Odometry Distance Displacement

class Sensor:
    def __init__(self, robot: Robot, map: UsimMap, sensor_parameters: SensorSettings = SensorSettings()) -> None:
        self.robot = robot
        self.last_odom_state = np.array([self.robot.data['theta'][-1], self.robot.data['x'][-1],
                                         self.robot.data['y'][-1]])
        self.last_robot_state = self.last_odom_state.copy()
        self.map: UsimMap = map

        self.camera_fov = sensor_parameters.camera_fov
        self.camera_range = sensor_parameters.camera_range
        self.lidar_range = sensor_parameters.lidar_range
        self.lidar_angular_resolution = sensor_parameters.lidar_angular_resolution

        self.lidar_range_noise_sigma = sensor_parameters.lidar_range_noise_sigma
        self.lidar_angular_noise_sigma = sensor_parameters.lidar_angular_noise_sigma
        self.odometry_angular_noise_sigma = sensor_parameters.odometry_angular_noise_sigma
        self.odometry_r_noise_sigma = sensor_parameters.odometry_r_noise_sigma

        self.lidar_angles = np.linspace(0.0, 360.0, num=self.lidar_angular_resolution, endpoint=False)

    def sample_sensors(self) -> tuple[np.ndarray, list[tuple[int, np.ndarray]], np.ndarray]:
        robot_state = np.array([self.robot.data['theta'][-1], self.robot.data['x'][-1],
                       self.robot.data['y'][-1]])

        odometry = self.odometry_measurements(robot_state)
        landmarks = self.camera_measurements(robot_state)
        lidar = self.lidar_measurements(robot_state)

        return (odometry, landmarks, lidar)

    def odometry_measurements(self, robot_state: np.ndarray) -> np.ndarray:
        # Add relative noise to r and delta_theta measurements
        diff = robot_state - self.last_robot_state

        r_noise, delta_theta_noise = np.random.multivariate_normal(
            [0, 0],
            np.square(np.diag([self.odometry_r_noise_sigma, self.odometry_angular_noise_sigma]))
        )
        delta_theta = np.mod(diff[0] + 180, 360) - 180
        delta_theta_noisy = delta_theta*(1 + delta_theta_noise)
        
        delta_pos_noisy = m.R(delta_theta*delta_theta_noise) @ diff[1:3]*(1 + r_noise)

        odom = np.array([self.last_odom_state[0] + delta_theta_noisy,
                         self.last_odom_state[1] + delta_pos_noisy[0],
                         self.last_odom_state[2] + delta_pos_noisy[1]])

        self.last_odom_state = odom.copy()
        self.last_robot_state = robot_state.copy()

        odom[0] = np.deg2rad(odom[0])

        return odom

    def camera_measurements(self, robot_state: np.ndarray) -> list[tuple[int, np.ndarray]]:
        observed_landmarks = []

        robot_position = np.array([robot_state[1], robot_state[2]])
        robot_heading = robot_state[0]

        # Check which landmarks are within the robot's camera field of view.
        for landmark_id, landmark_state in self.map.landmarks.items():
            landmark_position = landmark_state[0:2]
            landmark_orientation = landmark_state[2]
            
            # Offset the landmark to the robot's coordinate frame
            landmark_relative_position = landmark_position - robot_position
            landmark_relative_angle = np.rad2deg(np.arctan2(landmark_relative_position[1], landmark_relative_position[0])) - robot_heading
            landmark_relative_distance = np.linalg.norm(landmark_relative_position)

            landmark_relative_angle = landmark_relative_angle + 360 if landmark_relative_angle < -180 else landmark_relative_angle
            landmark_relative_angle = landmark_relative_angle - 360 if landmark_relative_angle > 180 else landmark_relative_angle

            landmark_relative_orientation = landmark_orientation - robot_heading

            # Determines if the landmark is in the camera's field of view and range
            is_in_fov = landmark_relative_angle > - self.camera_fov / 2 and landmark_relative_angle < self.camera_fov / 2
            is_in_range = landmark_relative_distance < self.camera_range
            if is_in_fov and is_in_range:
                observed_landmarks += [(landmark_id, np.array([landmark_relative_distance, np.deg2rad(landmark_relative_angle), np.deg2rad(landmark_relative_orientation)]))]
        
        return observed_landmarks
        
    def lidar_measurements(self, robot_state: np.ndarray) -> np.ndarray:
        ranges = np.zeros_like(self.lidar_angles)
        theta, x, y = robot_state

        wall_direction = np.array([0, 0])
        
        wall_vector = []

        # Prepare wall information
        for line in self.map.lines:
            xw0, yw0, xw1, yw1 = line
            # Build a wall direction vector, which is orthogonal to the orthogonal line
            wall_direction[0] = xw1-xw0
            wall_direction[1] = yw1-yw0
            wall_direction = wall_direction / np.linalg.norm(wall_direction)
            # Build a normal vector
            orthogonal_direction = np.empty_like(wall_direction)
            orthogonal_direction[0] = -wall_direction[1]
            orthogonal_direction[1] = wall_direction[0]

            # Vectors from robot position to line points
            xw0 = xw0 - x
            xw1 = xw1 - x
            yw0 = yw0 - y
            yw1 = yw1 - y

            # Determine the line position along chosen normal
            c = orthogonal_direction[0] * xw0 + orthogonal_direction[1] * yw0

            # Determine the intersection point
            xi = orthogonal_direction[0] * c
            yi = orthogonal_direction[1] * c

            angle_0 = np.arctan2(yw0, xw0) - np.arctan2(yi, xi)
            angle_1 = np.arctan2(yw1, xw1) - np.arctan2(yi, xi)

            angle_0 = np.mod(angle_0 + np.pi, 2*np.pi) - np.pi
            angle_1 = np.mod(angle_1 + np.pi, 2*np.pi) - np.pi

            wall_vector += [(np.linalg.norm(np.array([xi, yi])), np.arctan2(yi, xi), \
                             min(angle_0, angle_1), max(angle_0, angle_1))]


        for idx, angle in enumerate(self.lidar_angles):
            lidar_angle = np.deg2rad(angle + theta)
            ranges[idx] = np.inf

            for wall in wall_vector:
                r, phi, min_angle, max_angle = wall
                angle_to_orthogonal = lidar_angle - phi

                angle_to_orthogonal = np.mod(angle_to_orthogonal + np.pi, 2*np.pi) - np.pi

                if angle_to_orthogonal < min_angle or angle_to_orthogonal > max_angle:
                    continue

                if np.cos(angle_to_orthogonal) == 0:
                    continue

                distance_wall = r / np.cos(angle_to_orthogonal)
                if distance_wall < 0 or distance_wall > self.lidar_range:
                    continue

                ranges[idx] = min(ranges[idx], distance_wall)

            if ranges[idx] == np.inf:
                ranges[idx] = 0     # means out of range
        
        return ranges 