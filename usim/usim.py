from dataclasses import dataclass
from time import sleep

import matplotlib.pyplot as plt
import numpy as np

from .map import Map, load_map
from .robot import Robot, RobotSettings, RobotData
from .sensor import Sensor, SensorData

@dataclass
class SimulationData:
    sampling_time: float
    sensors: SensorData
    robot_pose: RobotData
    map: Map

if __name__=="__main__":
    sampling_time = 1e-3
    sensor_sampling_time = 1e-1
    simulation_time = 5

    robot = Robot(RobotSettings(), [0, 0, 0])
    map = load_map("map1.map")
    sensor = Sensor(robot, map)

    simulation_mode = 'interactive'

    if simulation_mode == 'interactive':
        ...
    elif simulation_mode == 'plot_data':
        ...

    sensor_data = []

    for i in range(int(simulation_time / sampling_time)):
        robot.simulation_step(0, 0.1, sampling_time)

        if i % int(sensor_sampling_time / sampling_time) == 0:
            sensor_data += [sensor.sample_sensors()]

    plt.ion()
    plt.figure(1)
    robot_trajectory_x = []
    robot_trajectory_y = []
    for data_point in sensor_data:
        odometry, landmarks, lidar = data_point

        heading, x, y = odometry
        measurement_cloud_x = []
        measurement_cloud_y = []
        robot_trajectory_x += [x]
        robot_trajectory_y += [y]
        for index, distance in enumerate(lidar):
            if distance == 0:
                continue

            angle = heading + index

            measurement_cloud_x += [x + distance * np.cos(np.deg2rad(angle))]
            measurement_cloud_y += [y + distance * np.sin(np.deg2rad(angle))]

        plt.scatter(measurement_cloud_x, measurement_cloud_y)
    plt.scatter(robot_trajectory_x, robot_trajectory_y)
    plt.show()