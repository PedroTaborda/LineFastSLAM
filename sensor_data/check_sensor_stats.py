import sensor_data.sensor_data as sd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default='sim0.xz')
args = parser.parse_args()
data = sd.load_sensor_data(args.file)
to = np.array([k[0] for k in data.odometry])
tc = np.array([k[0] for k in data.camera])
tl = np.array([k[0] for k in data.lidar])

plt.figure()
plt.plot(1e9/(np.diff(to)))
plt.ylabel("Odometry Frequency Hz")
plt.xlabel("Sample")
ylim = list(plt.ylim())
plt.ylim((0, min(31, ylim[1])))

plt.figure()
plt.plot(1e9/(np.diff(tc)))
plt.ylabel("Camera Frequency Hz")
plt.xlabel("Sample")
ylim = list(plt.ylim())
plt.ylim((0, min(31, ylim[1])))

plt.figure()
plt.plot(1e9/(np.diff(tl)))
plt.ylabel("Lidar Frequency Hz")
plt.xlabel("Sample")
ylim = list(plt.ylim())
plt.ylim((0, min(31, ylim[1])))

plt.show()
