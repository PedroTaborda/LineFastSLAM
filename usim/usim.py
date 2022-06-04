from __future__ import annotations
from dataclasses import dataclass
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_w,
    K_s,
    K_a,
    K_d,
    K_x,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

from .umap import UsimMap, load_map
from .robot import Robot, RobotSettings, RobotData
from .sensor import Sensor
import sensor_data.sensor_data as sd

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default="map1.map")
    parser.add_argument("--sampling-time", type=float, default=0.1)
    parser.add_argument("--sensor-sampling-time", type=float, default=0.1)

    args = parser.parse_args()

    sampling_time = args.sampling_time
    sensor_sampling_time = args.sensor_sampling_time
    map_file = args.map

    robot = Robot(RobotSettings(), [0, 0, 0])
    map = load_map(map_file)
    sensor = Sensor(robot, map)
    odometry_data = []
    camera_data = []
    lidar_data = []

    # Initialize pygame
    pg.init()

    # Define constants for the screen width and height
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 600
    TARGET_FPS = 60

    # Create the screen object
    # The size is determined by the constant SCREEN_WIDTH and SCREEN_HEIGHT
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    running = True
    linear_velocity = 0
    max_linear_velocity = 0.5
    angular_velocity = 0
    max_angular_velocity = 0.4
    velocity_steps = 10
    last_frame_time = time.time()
    last_simulation_step_time = time.time()
    last_simulation_measurement_time = time.time()
    while running:
        # Do the simulation of the robot
        if time.time() - last_simulation_step_time >= sampling_time:
            last_simulation_step_time = time.time()
            robot.simulation_step(angular_velocity, linear_velocity, sampling_time)

        if time.time() - last_simulation_measurement_time >= sensor_sampling_time:
            last_simulation_measurement_time = time.time()
            odometry, camera, lidar = sensor.sample_sensors()
            timestamp_ns = time.time_ns()
            odometry_data += [(timestamp_ns, odometry)]
            camera_data += [(timestamp_ns, camera)]
            lidar_data += [(timestamp_ns, lidar)]

        # Check if we have to draw
        current_frame_time = time.time()
        if current_frame_time - last_frame_time < 1 / TARGET_FPS:
            continue

        # Draw the frame and handle events
        last_frame_time = current_frame_time
        for event in pg.event.get():
            # Did the user hit a key?
            if event.type == KEYDOWN:
                # Was it the Escape key? If so, stop the loop.
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_UP or event.key == K_w:
                    if linear_velocity < 0:
                        linear_velocity = 0
                    else:
                        linear_velocity = min(max_linear_velocity, linear_velocity + max_linear_velocity/velocity_steps)
                elif event.key == K_DOWN or event.key == K_x:
                    if linear_velocity > 0:
                        linear_velocity = 0
                    else:
                        linear_velocity = max(-max_linear_velocity, linear_velocity - max_linear_velocity/velocity_steps)
                elif event.key == K_LEFT or event.key == K_a:
                    if angular_velocity < 0:
                        angular_velocity = 0
                    else:
                        angular_velocity = min(max_angular_velocity, angular_velocity + max_angular_velocity/velocity_steps)
                elif event.key == K_RIGHT or event.key == K_d:
                    if angular_velocity > 0:
                        angular_velocity = 0
                    else:
                        angular_velocity = max(-max_angular_velocity, angular_velocity - max_angular_velocity/velocity_steps)
                elif event.key == K_s:
                    linear_velocity = 0
                    angular_velocity = 0

            # Did the user click the window close button? If so, stop the loop.
            if event.type == QUIT:
                print("\n")
                pg.quit()
                running = False

        if not running:
            break

        screen.fill((255, 255, 255))
        map_surf = pg.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        map_surf.fill((255, 255, 255))

        for line in map.lines:
            xw0, yw0, xw1, yw1 = line

            xw0 = (xw0 - robot.data['x'][-1] + 5) / 10 * SCREEN_WIDTH
            xw1 = (xw1 - robot.data['x'][-1] + 5) / 10 * SCREEN_WIDTH
            yw0 = (-(yw0 - robot.data['y'][-1]) + 5) / 10 * SCREEN_HEIGHT
            yw1 = (-(yw1 - robot.data['y'][-1]) + 5) / 10 * SCREEN_HEIGHT

            pg.draw.line(map_surf, (0, 0, 0), (xw0, yw0), (xw1, yw1), width=3)

        screen.blit(map_surf, (0, 0))

        # Create the object for the turtlebot
        turtle = pg.Surface((55, 25), pg.SRCALPHA)
        turtle.fill((0, 0, 0))
        rect = turtle.get_rect()
        turtle = pg.transform.rotate(turtle, robot.data['theta'][-1])
        print(f"[theta, x, y] = [{robot.data['theta'][-1] % 360.0:06.2f}, {robot.data['x'][-1]:05.2f}, {robot.data['y'][-1]:05.2f}]", end='\r')
        surf_center = (
            (SCREEN_WIDTH-turtle.get_width())/2,
            (SCREEN_HEIGHT-turtle.get_height())/2
        )
        screen.blit(turtle, surf_center)
        pg.display.flip()

    poses = np.array(list(zip(robot.data['x'], robot.data['y'], robot.data['theta'])))
    sim_data = sd.SimulationData(sampling_time=sampling_time, robot_pose=poses, map=map)
    sensor_data_object = sd.SensorData(odometry=odometry_data, lidar=lidar_data, camera=camera_data,
                                       comment='From microsimulator', sim_data=sim_data)

    files = os.listdir(sd.DEFAULT_SENSOR_DATA_DIR)
    num=0
    format = lambda num: f'sim{num}.xz'
    while format(num) in files:
        num+=1
    sd.save_sensor_data(sensor_data_object, format(num))

def plot_frame(frame, sensor_data):
    plt.figure(1)
    plt.clf()
    data_point = sensor_data[frame]
    odometry, landmarks, lidar = data_point

    heading, x, y = odometry
    measurement_cloud_x = []
    measurement_cloud_y = []
    for index, distance in enumerate(lidar):
        if distance == 0:
            continue

        angle = heading + index

        measurement_cloud_x += [x + distance * np.cos(np.deg2rad(angle))]
        measurement_cloud_y += [y + distance * np.sin(np.deg2rad(angle))]

    plt.scatter(measurement_cloud_x, measurement_cloud_y)
    plt.show()
