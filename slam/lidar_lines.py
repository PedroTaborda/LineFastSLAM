import os
import copy
import shutil

import numpy as np
import matplotlib.pyplot as plt

from slam.ransac import RANSAC, RansacModel
from visualization_utils.mpl_video import to_video

class StraightLineModel(RansacModel):
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.direction: np.ndarray = None

    def fit(self, data_points: np.ndarray):
        """
        Args:
            data_points: (N, 3) array of N data points with [x1, x2, 1] coordinates.

        Returns:
            None

        Side Effects:
            Sets self.direction, such that self.inlier(data_points) returns a 
            logical array of the inliers in data_points 
        """
        vals, vecs = np.linalg.eigh(data_points.T @ data_points) 
        self.direction = vecs[:, 0]

    def inliers(self, data_points):
        projections = data_points @ self.direction
        return data_points[np.abs(projections) < self.threshold]

    def idxoutliers(self, data_points):
        projections = data_points @ self.direction
        # print(f"{self.direction = }")
        return np.abs(projections) > self.threshold


def plot_line(ax, rh, th, label, color, plot: bool = True):
    direction = np.array([np.sin(th), -np.cos(th)])
    x0 = rh*np.array([np.cos(th), np.sin(th)])
    p = np.array([x0 - rh*2*direction, x0 + rh*2*direction])
    # print(f"rh, th = {rh}, {th}")
    # print(f"{p=}")
    if plot:
        ax.plot([0, rh*np.cos(th)], [0, rh*np.sin(th)], "--", color=color)
        ax.plot(p[:, 0], p[:, 1], label=label, color=color)
    return {
        "x": [0, rh*np.cos(th)],
        "y": [0, rh*np.sin(th)],
        "kwargs": {
            "linestyle": "--",
            "color": color
        }
    }, {
        "x": p[:, 0],
        "y": p[:, 1],
        "kwargs": {
            "label": label,
            "color": color
        }
    }
def identify_lines(scan: np.ndarray, plot_ax: plt.Axes = None, demo: bool = False, demo_dir: str = "ransac_demo") -> np.ndarray:
    """
    Identify lines in a lidar scan.

    Args:
        scan: The lidar scan data.

    Returns:
        The identified lines as a list of tuples of (rh, th).
    """
    lines = []
    
    good = np.where(scan > 0.01)[0] 
    angles = good.astype(float)*np.pi/180
    cleaned_scan = scan[good]
    xypoints = (np.array([np.cos(angles), np.sin(angles)]) * cleaned_scan).T
    xypoints = np.block([xypoints, np.ones_like([xypoints[:, 0]]).T])
    model = StraightLineModel()

    minp = 15
    noise_rejection_factor = 20

    if demo:
        fig, ax = plt.subplots(1, 1)
    ax: plt.Axes
    if os.path.isdir(demo_dir):
        shutil.rmtree(demo_dir)
    if not os.path.isdir(demo_dir):
        os.mkdir(demo_dir)

    demo_xlims = None
    demo_ylims = None
    lines_to_plot = []
    prev_xypoints = copy.copy(xypoints)

    def save_frame(demo_frame_lst = [0], clear=True):
        demo_frame = demo_frame_lst[0]
        ax.set_xlim(demo_xlims)
        ax.set_ylim(demo_ylims)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        plt.savefig(os.path.join(demo_dir, f"{demo_frame}.png"), dpi=500)
        demo_frame_lst[0] = demo_frame + 1
        if clear:
            ax.cla()

    def get_line(data_points, t):
        # Identify lines in the lidar scan.
        best_model, best_inliers = RANSAC(data_points, model, 2, t=t, k=500)
        a, b, c = best_model.direction
        # print(f"{best_model.direction=}")
        th = np.arctan2(b, a)
        rh = -c/np.sqrt(a**2 + b**2)
        if rh < 0:
            rh, th = -rh, th + np.pi
        th = np.mod(th + np.pi, 2*np.pi) - np.pi
        return (rh, th), best_inliers, best_model

    if demo:
        ax.plot(xypoints[:, 0], xypoints[:, 1], "o")
        demo_xlims = ax.get_xlim()
        demo_ylims = ax.get_ylim()
        save_frame()

    i = 0
    while xypoints.shape[0] >= minp:
        # find a line.
        line, inliers, model = get_line(xypoints, min(3*minp, xypoints.shape[0])/xypoints.shape[0])
        if len(inliers) < minp: # minimum number of points to consider a line.
            break
        lines.append(line)
        # remove inliers from xypoints
        model.threshold *= noise_rejection_factor # increase threshold outlier rejection to remove noise around detected line
        prev_xypoints = xypoints
        # print(xypoints)
        # print(model.idxoutliers(xypoints))
        xypoints = xypoints[model.idxoutliers(xypoints)]
        model.threshold /= noise_rejection_factor # decrease threshold for next line
        if plot_ax is not None:
            plot_ax.plot(inliers[:, 0], inliers[:, 1], "o", markersize=3, color=f'C0{i+1}')
        lines_to_plot.append({
            "x": inliers[:, 0],
            "y": inliers[:, 1],
            "kwargs": {
                "marker": "o",
                "markersize": 3,
                "color": f'C0{i+1}'
            }
        })
        if demo:
            ax.plot(prev_xypoints[:, 0], prev_xypoints[:, 1], "o")
            l1, l2 = plot_line(ax, *line, "", f'C0{i+1}', plot=False)
            lines_to_plot.append(l1)
            lines_to_plot.append(l2)
            for line_to_plot in lines_to_plot:
                ax.plot(line_to_plot["x"], line_to_plot["y"], **line_to_plot["kwargs"])
            save_frame()
            ax.plot(xypoints[:, 0], xypoints[:, 1], "o")
            for line_to_plot in lines_to_plot:
                ax.plot(line_to_plot["x"], line_to_plot["y"], **line_to_plot["kwargs"])
            save_frame()
        i += 1

    if demo:
        plt.close(ax.get_figure())
    return lines

if __name__ == "__main__":
    # Read a SensorData object from a file to extract lidar scan data to test the lidar scanner.
    from sensor_data.sensor_data import SensorData, load_sensor_data
    import matplotlib.pyplot as plt

    plt.ion()
    sd: SensorData = load_sensor_data("corridor-w-light.xz")
    imgs_dir = os.path.join('images', 'ransac')
    if not os.path.isdir(imgs_dir):
        os.mkdir(imgs_dir)

    video_fig, video_ax = plt.subplots(1, 1)

    frame = 50
    identify_lines(sd.lidar[frame][1], plot_ax=video_ax, demo=True, demo_dir=os.path.join("data","ransac_demo_frame50"))
    frame = 500
    identify_lines(sd.lidar[frame][1], plot_ax=video_ax, demo=True, demo_dir=os.path.join("data","ransac_demo_frame500"))

    for frame, instant in enumerate(sd.lidar):
        ex_scan = instant[1]
        #ex_scan = sd.lidar[500][1] # index [i][0] contains the timestamp
        ex_scan[ex_scan > 3.49] = 0.0

        good = np.where(ex_scan > 0.01)[0] 
        angles = good.astype(float)*np.pi/180
        cleaned_scan = ex_scan[good]
        xypoints = (np.array([np.cos(angles), np.sin(angles)]) * cleaned_scan).T

        video_ax.cla()      
        video_ax.plot(xypoints[:, 0], xypoints[:, 1], "o", label="Lidar scan")
        lines = identify_lines(ex_scan, plot_ax=video_ax, demo=False)

        
        # Plot the lidar scan and the identified lines.
        for idx, (rh, th) in enumerate(lines):
            plot_line(video_ax, rh, th, label = f"line {idx}", color=f'C0{idx+3}')

        video_ax.set_title(f"Frame {frame}")
        plt.pause(0.01)
        plt.savefig(os.path.join(imgs_dir,f"lidar_scan_{instant[0]}.png"))


    to_video(imgs_dir, "lidar_scan.mp4")
