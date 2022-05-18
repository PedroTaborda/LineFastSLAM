import os

import imageio
import numpy as np
import matplotlib.pyplot as plt

import rosbags.rosbag1
import rosbags.serde

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

def animate_laser_scans(scans, save_dir = None):
    angles = np.linspace(scans[0].angle_min, scans[0].angle_max, num=scans[0].ranges.shape[0])
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    lines = [ax.plot([0, angles[i]], [0, scans[0].ranges[i]], 'C00') for i in range(angles.shape[0])]
    ax.grid('on')
    plt.show(block=False)
    for idx, scan in enumerate(scans):
        for i in range(angles.shape[0]):
            lines[i][0].set_data([0, angles[i]], [0, scan.ranges[i]])
        fig.canvas.draw()
        fig.canvas.flush_events()

        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, f'{idx:05d}.png'))

def to_video(image_dir, video_name: str = 'simulation.mp4', step_size_plot: float=1/10):
    """ Compile every frame into a video and save it to a file.
    """
    print('Saving video...')
    fps = int(1/step_size_plot)
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            images.append(os.path.join(image_dir, filename))
    images.sort()
    writer = imageio.get_writer(video_name, fps=fps)
    for image in images:
        writer.append_data(imageio.imread(image))
        os.remove(image)
    writer.close()

if __name__ == '__main__':
    image_dir = os.path.join('images')
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)
    animate_laser_scans(laser_scans[1300:], save_dir = image_dir)
    to_video(image_dir, video_name='lidar1_10fps.mp4')
    animate_laser_scans(laser_scans[:400], save_dir = image_dir)
    to_video(image_dir, video_name='lidar2_10fps.mp4')