import rosbags.rosbag1
import rosbags.serde

import numpy as np
import matplotlib.pyplot as plt

from timeseries import Timeseries

laser_scans = []
with rosbags.rosbag1.Reader('1.bag') as reader:
    print(list(reader.topics.keys()))
    connections_laser = []
    for x in reader.connections:
        if x.topic == '/scan':
            connections_laser.append(x)
    for connection, timestamp, rawdata in reader.messages(connections=connections_laser):
        msg = rosbags.serde.deserialize_cdr(rosbags.serde.ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype)
        laser_scans.append(msg)

def plot_laser_scan(scan):
    angles = np.linspace(scan.angle_min, scan.angle_max, num=scan.ranges.shape[0])
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    lines = [ax.plot([0, angles[i]], [0, scan.ranges[i]], 'C00') for i in range(angles.shape[0])]
    ax.grid('on')
    plt.show()

def animate_laser_scans(scans):
    angles = np.linspace(scans[0].angle_min, scans[0].angle_max, num=scans[0].ranges.shape[0])
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    lines = [ax.plot([0, angles[i]], [0, scans[0].ranges[i]], 'C00') for i in range(angles.shape[0])]
    ax.grid('on')
    plt.show(block=False)
    for scan in scans:
        for i in range(angles.shape[0]):
            lines[i][0].set_data([0, angles[i]], [0, scan.ranges[i]])
        fig.canvas.draw()
        fig.canvas.flush_events()


