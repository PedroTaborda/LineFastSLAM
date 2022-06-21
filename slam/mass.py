import multiprocessing as mp
import os
import pickle
import copy

import slam.fastslam as fs
import slam.offline as offline
import sensor_data.sensor_data as sd

def file_name(settings: fs.FastSLAMSettings) -> str:
    """
    Generate a file name for the given settings.
    """
    return settings.hash_str()

def perform_slam(sensor_and_settings_obj: tuple[sd.SensorData, fs.FastSLAMSettings]):
    """
    Perform a single FastSLAM run with the given settings.
    """
    sensor_data, settings = sensor_and_settings_obj
    res = offline.slam_sensor_data(
        sensor_data, 
        slam_settings=settings,
        images_dir=None,
        realtime=False,
        show_images=False,
        stats_iter_size=1
    )
    return res

def slam_batch(settings: list[fs.FastSLAMSettings], sensor_data: sd.SensorData, 
                repeats: int = 5, pool_processes: int = 4, results_dir = 'slammed') -> list[fs.FastSLAM]:
    """
    Run FastSLAM on a batch of settings, 'repeats' times per settings object.
    """
    if not repeats:
        expanded_settings = settings
    else:
        expanded_settings = []
        for s in settings:
            for i in range(repeats):
                new_settings = copy.copy(s)
                new_settings.rng_seed = i
                expanded_settings.append(s)

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    n_processed = 0
    to_process = []
    for s in expanded_settings:
        rel_path = os.path.join(results_dir, file_name(s))
        if not os.path.exists(rel_path):
            to_process.append((sensor_data, s))
        else:
            n_processed += 1

    print(f"Processing {len(to_process)} settings, {n_processed} already processed.")

    pool = mp.Pool(processes=pool_processes)
    results = pool.map(perform_slam, to_process)

    for s, res in zip(to_process, results):
        rel_path = os.path.join(results_dir, file_name(s[1]))
        with open(rel_path, 'wb') as f:
            pickle.dump(res, f)

    res_ret = []
    for settings in expanded_settings:
        rel_path = os.path.join(results_dir, file_name(settings))
        with open(rel_path, 'rb') as f:
            res_ret.append(pickle.load(f))

    return res_ret



if __name__ == "__main__":
    settings_collection = [
        fs.FastSLAMSettings(
            num_particles=n,
        ) for n in [5, 10, 20, 40, 80]
    ]

    sensor_data = sd.load_sensor_data('sim0.xz')

    res = slam_batch(settings_collection, sensor_data, repeats=5, pool_processes=4)