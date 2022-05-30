import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider

import numpy as np

from .sensor_data import SensorData, load_sensor_data

class SensorDataViewer:
    def __init__(self, sensor_data: SensorData, fig_num: int) -> None:
        self.sensor_data: SensorData = sensor_data
        self.fig, self.axs = plt.subplots(2, 1, num=fig_num)

        self.lidar_ax, self.pos_ax = self.axs
        
        self.time_slider: Slider = Slider(
            self.lidar_ax,
            label   = 'Time (s)', 
            valmin  = 0, 
            valmax  = self._determine_data_length(sensor_data), 
            valinit = 0,
            valstep = self._iteration_length(sensor_data)
        )

        def _on_time_slider_change(val):
            self.on_time_slider_change(val)

        self.time_slider.on_changed(_on_time_slider_change)

        # Plot the odometry
        theta, x, y = self.sensor_data.odometry[0][1]
        self.odometry_line = plt.scatter(x, y, c='r', marker=(3, 0, theta - 90))

        # Plot the camera data
        camera_display_data = []
        camera_x_data = []
        camera_y_data =[]
        for id, landmark in self.sensor_data.camera[0][1]:
            phi, r = landmark
            camera_display_data += [id]
            camera_x_data += [x + r*np.cos(np.deg2rad(theta + phi))]
            camera_y_data += [y + r*np.sin(np.deg2rad(theta + phi))]

        self.camera_line = plt.scatter(camera_x_data, camera_y_data, c='b', marker='o', data=camera_display_data)

        # Plot the lidar data
        lidar_x_data = []
        lidar_y_data = []
        for angle in range(len(self.sensor_data.lidar[0][1])):
            if self.sensor_data.lidar[0][1][angle] == 0.0:
                continue
            lidar_x_data += [x + self.sensor_data.lidar[0][1][angle]*np.cos(np.deg2rad(theta + angle))]
            lidar_y_data += [y + self.sensor_data.lidar[0][1][angle]*np.sin(np.deg2rad(theta + angle))]

        self.lidar_line = plt.scatter(lidar_x_data, lidar_y_data, c='k', marker='.')

        plt.tight_layout()

    def on_time_slider_change(self, val):
        nearest_odometry_sample = min(int(val / self.odometry_sampling_time), len(self.sensor_data.odometry) - 1)
        nearest_lidar_sample = min(int(val / self.lidar_sampling_time), len(self.sensor_data.lidar) - 1)
        nearest_camera_sample = min(int(val / self.camera_sampling_time), len(self.sensor_data.camera) - 1)

        self.pos_ax.clear()

        # Plot the odometry
        theta, x, y = self.sensor_data.odometry[nearest_odometry_sample][1]
        self.odometry_line = plt.scatter(x, y, c='r', marker=(3, 0, theta - 90))

        # Plot the camera data
        camera_display_data = []
        camera_x_data = []
        camera_y_data =[]
        for id, landmark in self.sensor_data.camera[nearest_camera_sample][1]:
            phi, r = landmark
            camera_display_data += [id]
            camera_x_data += [x + r*np.cos(np.deg2rad(theta + phi))]
            camera_y_data += [y + r*np.sin(np.deg2rad(theta + phi))]

        self.camera_line = plt.scatter(camera_x_data, camera_y_data, c='b', marker='o', data=camera_display_data)

        # Plot the lidar data
        lidar_x_data = []
        lidar_y_data = []
        for angle in range(len(self.sensor_data.lidar[nearest_lidar_sample][1])):
            if self.sensor_data.lidar[nearest_lidar_sample][1][angle] == 0.0:
                continue
            lidar_x_data += [x + self.sensor_data.lidar[nearest_lidar_sample][1][angle]*np.cos(np.deg2rad(theta + angle))]
            lidar_y_data += [y + self.sensor_data.lidar[nearest_lidar_sample][1][angle]*np.sin(np.deg2rad(theta + angle))]

        self.lidar_line = plt.scatter(lidar_x_data, lidar_y_data, c='k', marker='.')
        self.pos_ax.set_xlim((-10, 10))
        self.pos_ax.set_ylim((-10, 10))

    def show(self):
        plt.show()

    def _determine_data_length(self, data: SensorData) -> float:
        scale_factor = 1e-9

        odometry_length = (data.odometry[-1][0] - data.odometry[0][0]) * scale_factor
        lidar_length = (data.lidar[-1][0] - data.lidar[0][0]) * scale_factor
        camera_length = (data.camera[-1][0] - data.camera[0][0]) * scale_factor

        return min((odometry_length, lidar_length, camera_length))

    def _iteration_length(self, data: SensorData) -> float:
        scale_factor = 1e-9
        
        self.odometry_sampling_time = (data.odometry[1][0] - data.odometry[0][0]) * scale_factor
        self.lidar_sampling_time = (data.lidar[1][0] - data.lidar[0][0]) * scale_factor
        self.camera_sampling_time = (data.camera[1][0] - data.camera[0][0]) * scale_factor

        return max((self.odometry_sampling_time, self.lidar_sampling_time, self.camera_sampling_time))

def view_sensor_data(sensor_data: SensorData, fig_num: int = 1) -> SensorDataViewer:
    return SensorDataViewer(sensor_data, fig_num)


if __name__=='__main__':
    data = load_sensor_data('sim0.xz')
    viewer = view_sensor_data(data)
    viewer.show()