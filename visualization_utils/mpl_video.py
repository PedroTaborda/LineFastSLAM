import os

import imageio


def to_video(image_dir, video_name: str = 'simulation.mp4', filename_termination = '.png', fps: float=10):
    """ Compile every frame into a video and save it to a file.
    """
    print(f'Saving video from images in "{image_dir}"...')
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith(filename_termination):
            images.append(os.path.join(image_dir, filename))
    images.sort()
    writer = imageio.get_writer(video_name, fps=fps)
    for image in images:
        writer.append_data(imageio.imread(image))
        os.remove(image)
    writer.close()
    print(f'Video saved to "{video_name}".')
