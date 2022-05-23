from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .robot import Robot
from .map import Map

@dataclass
class SensorSettings:
    # Sensor parameters
    camera_fov: float = 90                      # Field of view in degrees
    camera_range: float = 25                    # Maximum range that the camera can detect the Aruco Markers
    lidar_range: float = 3.5                    # Range in meters
    lidar_angular_resolution: float = 360       # Number of points in one full sweep

    # Sensor noise characterization
    lidar_range_noise_sigma: float = 0          # Standard Deviation of Lidar Range Measurements
    lidar_angular_noise_sigma: float = 0        # Standard Deviation of Lidar Angular Sweep Position
    odometry_angular_noise_sigma: float = 0     # Standard Deviation of Odometry Angular Position
    odometry_cartesian_noise_sigma: float = 0   # Standard Deviation of Odometry Cartesian Position

class Sensor:
    def __init__(self, robot: Robot, map: Map, sensor_parameters: SensorSettings = SensorSettings()) -> None:
        self.robot = robot
        self.map: Map = map

        self.camera_fov = sensor_parameters.camera_fov
        self.camera_range = sensor_parameters.camera_range
        self.lidar_range = sensor_parameters.lidar_range
        self.lidar_angular_resolution = sensor_parameters.lidar_angular_resolution

        self.lidar_range_noise_sigma = sensor_parameters.lidar_range_noise_sigma
        self.lidar_angular_noise_sigma = sensor_parameters.lidar_angular_noise_sigma
        self.odometry_angular_noise_sigma = sensor_parameters.odometry_angular_noise_sigma
        self.odometry_cartesian_noise_sigma = sensor_parameters.odometry_cartesian_noise_sigma

        self.lidar_angles = np.linspace(0.0, 360.0, num=self.lidar_angular_resolution, endpoint=False)

    def sample_sensors(self) -> tuple[np.ndarray, list[tuple[int, float]], np.ndarray]:
        robot_state = np.array([self.robot.data['theta'][-1], self.robot.data['x'][-1],
                       self.robot.data['y'][-1]])

        odometry = self.odometry_measurements(robot_state)
        landmarks = self.camera_measurements(robot_state)
        lidar = self.lidar_measurements(robot_state)

        return (odometry, landmarks, lidar)

    def odometry_measurements(self, robot_state: np.ndarray) -> np.ndarray:
        # Linearly add noise to the odometry measurements
        measured_angle = robot_state[0] + np.random.normal(0.0, self.odometry_angular_noise_sigma, 1)
        measured_x = robot_state[1] + np.random.normal(0.0, self.odometry_cartesian_noise_sigma, 1)
        measured_y = robot_state[2] + np.random.normal(0.0, self.odometry_cartesian_noise_sigma, 1)

        return np.array([measured_angle, measured_x, measured_y])

    def camera_measurements(self, robot_state: np.ndarray) -> list[tuple[int, float]]:
        observed_landmarks = []

        robot_position = np.array([robot_state[1], robot_state[2]])
        robot_heading = robot_state[0]

        # Check which landmarks are within the robot's camera field of view.
        for landmark_id, landmark_position in self.map.landmarks.items():
            # Offset the landmark to the robot's coordinate frame
            landmark_relative_position = landmark_position - robot_position
            landmark_relative_angle = np.rad2deg(np.arctan2(landmark_relative_position[1], landmark_relative_position[0])) - robot_heading

            # Determines if the landmark is in the camera's field of view and range
            is_in_fov = landmark_relative_angle > - self.camera_fov / 2 and landmark_relative_angle < self.camera_fov / 2
            is_in_range = np.linalg.norm(landmark_relative_position) < self.camera_range
            if is_in_fov and is_in_range:
                observed_landmarks += [(landmark_id, landmark_relative_angle)]
        
        return observed_landmarks
        
    def lidar_measurements(self, robot_state: np.ndarray):
        ranges = np.zeros_like(self.lidar_angles)
        theta, x, y = robot_state

        wall_direction = np.array([0, 0])
        
        wall_vector = []

        # Prepare wall information
        for line in self.map.lines:
            xw0, yw0, xw1, yw1 = line
            # Build the wall direction vector, which is orthogonal to the orthogonal line
            wall_direction[0] = xw1-xw0
            wall_direction[1] = yw1-yw0
            wall_direction = wall_direction / np.linalg.norm(wall_direction)
            # Build the orthogonal vector
            orthogonal_direction = np.empty_like(wall_direction)
            orthogonal_direction[0] = -wall_direction[1]
            orthogonal_direction[1] = wall_direction[0]

            # Transform the line points to the robot frame
            xw0 = xw0 - x
            xw1 = xw1 - x
            yw0 = yw0 - y
            yw1 = yw1 - y

            # Determine the line bias
            c = -orthogonal_direction[0] * xw0 - orthogonal_direction[1] * yw0

            # Determine the intersection point
            xi = - orthogonal_direction[0] / (np.linalg.norm(orthogonal_direction) ** 2) * c
            yi = - orthogonal_direction[1] / (np.linalg.norm(orthogonal_direction) ** 2) * c

            angle_0 = np.arctan2(yw0, xw0)
            angle_1 = np.arctan2(yw1, xw1)

            wall_vector += [(np.linalg.norm(np.array([xi, yi])), np.arctan2(yi, xi), \
                             min(angle_0, angle_1), max(angle_0, angle_1))]


        for idx, angle in enumerate(self.lidar_angles):
            ranges[idx] = np.inf

            for wall in wall_vector:
                r, phi, min_angle, max_angle = wall
                lidar_angle = (np.deg2rad(angle + theta) + np.pi) % (2 * np.pi) - np.pi

                if lidar_angle < min_angle or lidar_angle > max_angle:
                    continue

                angle_to_orthogonal = np.rad2deg(phi - lidar_angle)
                if (angle_to_orthogonal < -89 and angle_to_orthogonal > -91 ) or \
                   (angle_to_orthogonal > 89 and angle_to_orthogonal < 91):
                    continue

                distance_wall = r / np.cos(np.deg2rad(angle_to_orthogonal))
                if distance_wall < 0 or distance_wall > self.lidar_range:
                    continue

                ranges[idx] = min(ranges[idx], distance_wall)

            if ranges[idx] == np.inf:
                ranges[idx] = 0

        # Vetor Perpendicular a parede
        # <(vector da parede), (vector robot e ponto a descobir)> = <(xw1-xw0, yw1-yw0), (?x - 0, ?y - 0)> = 0 <=> ?x = yw1-yw0, ?y = -(xw1-xw0)

        # Para descobrir a interseptação entre o vetor perpendicular e a parede
        # 
        
        return ranges


            
        