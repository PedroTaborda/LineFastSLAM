import cv2
import imageio
from matplotlib import pyplot as plt
import numpy as np

import os

import sensor_data.sensor_data as sd

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

if __name__=="__main__":
    data = sd.rosbag_to_data('data/camera-unmeasured.bag')
    imgs = sd.rosbag_to_imgs('data/camera-unmeasured.bag')
    camera_matrix, distortion_coefficients = sd.rosbag_camera_info('data/camera-unmeasured.bag')

    save_dir = 'data/video_d_a/'
    plt.ion()
    for idx, image in enumerate(imgs):
        new_image, _ = sd.detect_landmarks(image, camera_matrix, distortion_coefficients)

        fig = plt.figure(0)
        plt.clf()
        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.pause(0.001)
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, f'{idx:05d}.png'))

    to_video(save_dir)

    '''
    plt.ion()
    
    for i in range(len(data.lidar)):
        plt.figure(0)
        plt.clf()
        point_x = []
        point_y = []
        point_color = []
        for angle, point in enumerate(data.lidar[i]):
            point_x += [point*np.cos(np.deg2rad(angle + 90))]
            point_y += [point*np.sin(np.deg2rad(angle + 90))]
            point_color += [point]

        plt.scatter(point_x, point_y, c=point_color)
        plt.xlim((-3.5, 3.5))
        plt.ylim((-3.5, 3.5))

        plt.figure(1)
        plt.clf()
        plt.imshow(imgs[i])

        plt.show()
        plt.pause(0.001)
    '''