from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

import sensor_data.sensor_data as sd
import slam.offline as off
import slam.map as sm

def plot_map(estimated_map: sm.Map, trajectory: list[tuple[int, np.ndarray]], sensor_data: sd.SensorData, ax: plt.Axes):
    t_traj = np.array([traj[0] for traj in trajectory]) 
    t_lid = (np.array([lid[0] for lid in sensor_data.lidar]) - sensor_data.lidar[0][0])*1e-9

    pc_plot_handle: plt.Line2D = ax.plot([], [], markersize=0.1, linestyle='', marker='.', c='#000000', zorder=-10)[0]

    i_traj = 0
    for t_lid, pc_raw in sensor_data.lidar:
        t_lid = (t_lid - sensor_data.lidar[0][0])*1e-9
        while i_traj < t_traj.shape[0] and t_traj[i_traj] < t_lid:
            i_traj += 1
        i_traj -= 1
        if i_traj < 0:
            i_traj = 0

        instant_traj, pose = trajectory[i_traj]
        off.plot_pc(pc_plot_handle, pc_raw, pose)

    estimated_map._draw(ax, color_ellipse='C01', color_p='C01', color_z='C01')
    ax.axis('equal')
    ax.relim()
    ax.autoscale_view(False, True, True)
