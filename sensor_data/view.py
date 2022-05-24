import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider

from .sensor_data import SensorData

class SensorDataViewer:
    def __init__(self, sensor_data: SensorData, fig_num: int) -> None:
        self.sensor_data: SensorData = sensor_data
        plt.figure(fig_num)
        self.fig, self.axs = plt.subplots(2, 1)

        self.lidar_ax, self.pos_ax = self.axs
        
        self.time_slider: Slider = Slider(
            self.lidar_ax,
            label   = 'Time (s)', 
            valmin  = 0, 
            valmax  = sensor_data.lidar.shape[0]*sensor_data.ts, 
            valinit = 0,
            valstep = sensor_data.ts
        )

        def _on_time_slider_change(val):
            self.on_time_slider_change(val)

        self.time_slider.on_changed(_on_time_slider_change)

    def on_time_slider_change(self, val):
        ...

def view_sensor_data(sensor_data: SensorData, fig_num: int = 1) -> SensorDataViewer:
    return SensorDataViewer(sensor_data, fig_num)


