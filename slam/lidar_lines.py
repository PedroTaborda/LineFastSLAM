import os
import numpy as np
import matplotlib.pyplot as plt

from slam.ransac import RANSAC, RansacModel
from visualization_utils.mpl_video import to_video

class StraightLineModel(RansacModel):
    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self.direction: np.ndarray = None

    def fit(self, data_points: np.ndarray):
        """
        Args:
            data_points: (N, 2) array of N data points with x and y coordinates.

        Returns:
            None

        Side Effects:
            Sets self.direction, such that self.inlier(x) returns 
                True if x is an inlier.
        """
        # print(f'[INFO] ({inspect.currentframe().f_code.co_name}) Fitting straight line model')
        # print(data_points)
        A = np.block([data_points, np.ones_like([data_points[:, 0]]).T])
        vals, vecs = np.linalg.eig(A.T @ A) 
        self.direction = vecs[:, np.argmin(vals)]
        # print(f'[INFO] ({inspect.currentframe().f_code.co_name}) Direction: {self.direction}')

    def inliers(self, data_points):
        projections = np.block([data_points, np.ones_like([data_points[:, 0]]).T]) @ self.direction
        # print(f'[INFO] ({inspect.currentframe().f_code.co_name}) Finding inliers')
        # print(np.block([data_points, np.ones_like([data_points[:, 0]]).T]) @ self.direction)
        # print(f"Inliers = {data_points[np.where(projections < self.threshold)].shape}")
        return data_points[np.where( np.abs(projections) < self.threshold)]

    def idxoutliers(self, data_points):
        projections = np.block([data_points, np.ones_like([data_points[:, 0]]).T]) @ self.direction
        return np.abs(projections) > self.threshold

def identify_lines(scan: np.ndarray, plot: bool = False) -> np.ndarray:
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
    model = StraightLineModel()

    minp = 15
    noise_rejection_factor = 10

    def get_line(data_points, t):
        # Identify lines in the lidar scan.
        best_model, best_inliers = RANSAC(data_points, model, 2, t=t, k=500)
        a, b, c = best_model.direction
        th = np.arctan2(b, a)
        rh = -c/np.sqrt(a**2 + b**2)
        if rh < 0:
            rh, th = -rh, th + np.pi
        return (rh, th), best_inliers


    i = 0
    while xypoints.shape[0] >= minp:
        # find a line.
        line, inliers = get_line(xypoints, min(3*minp, xypoints.shape[0])/xypoints.shape[0])
        if len(inliers) < minp: # minimum number of points to consider a line.
            break
        lines.append(line)
        # remove inliers from xypoints
        model.threshold *= noise_rejection_factor # increase threshold outlier rejection to remove noise around detected line
        xypoints = xypoints[model.idxoutliers(xypoints)]
        model.threshold /= noise_rejection_factor # decrease threshold for next line
        if plot:
            plt.plot(inliers[:, 0], inliers[:, 1], 'o', markersize=3, color=f'C0{i+3}')
        i += 1

    return lines

if __name__ == "__main__":
    # Read a SensorData object from a file to extract lidar scan data to test the lidar scanner.
    from sensor_data.sensor_data import SensorData, load_sensor_data
    import matplotlib.pyplot as plt

    plt.ion()
    plt.figure()
    sd: SensorData = load_sensor_data("corridor-w-light.xz")
    imgs_dir = os.path.join('images', 'ransac')
    if not os.path.isdir(imgs_dir):
        os.mkdir(imgs_dir)

    for instant in sd.lidar:
        ex_scan = instant[1]
        #ex_scan = sd.lidar[500][1] # index [i][0] contains the timestamp
        ex_scan[ex_scan > 3.49] = 0.0

        good = np.where(ex_scan > 0.01)[0] 
        angles = good.astype(float)*np.pi/180
        cleaned_scan = ex_scan[good]
        xypoints = (np.array([np.cos(angles), np.sin(angles)]) * cleaned_scan).T

        plt.clf()        
        plt.plot(xypoints[:, 0], xypoints[:, 1], "o", label="Lidar scan")
        lines = identify_lines(ex_scan, plot=True)

        def plot_line(rh, th, label, color):
            direction = np.array([np.sin(th), -np.cos(th)])
            x0 = rh*np.array([np.cos(th), np.sin(th)])
            p = np.array([x0 - rh*2*direction, x0 + rh*2*direction])
            print(f"rh, th = {rh}, {th}")
            print(f"{p=}")
            plt.plot(p[:, 0], p[:, 1], label=label, color=color)
        
        # Plot the lidar scan and the identified lines.
        for idx, (rh, th) in enumerate(lines):
            plt.plot([0, rh*np.cos(th)], [0, rh*np.sin(th)], "--", color=f'C0{idx+3}')
            plot_line(rh, th, label = f"line {idx}", color=f'C0{idx+3}')

        plt.pause(0.01)
        plt.savefig(os.path.join(imgs_dir,f"lidar_scan_{instant[0]}.png"))
        plt.show()

    to_video(imgs_dir, "lidar_scan.mp4")
