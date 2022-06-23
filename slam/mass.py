import multiprocessing as mp
import os
import pickle
import copy
from slam.action_model import ActionModelSettings

import slam.fastslam as fs
import slam.offline as offline
import sensor_data.sensor_data as sd

if not os.path.isdir('data'):
    os.mkdir('data')

def file_name(settings: fs.FastSLAMSettings, sensor_data: sd.SensorData) -> str:
    """
    Generate a file name for the given settings.
    """
    return settings.hash_str() + sensor_data.hash_str()

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
        stats_iter_size=1,
        profile=False
    )
    return res

def slam_batch(settings: list[fs.FastSLAMSettings], sensor_data: sd.SensorData, 
                repeats: int = 2, pool_processes: int = 4, results_dir = 'slammed') -> list[fs.FastSLAM]:
    """
    Run FastSLAM on a batch of settings, 'repeats' times per settings object.
    Returns a list of lists of fs.SLAMResult objects.
    Each list contains the results for one settings object for several seeds.

    These results are stored in files, indexed by the settings object hash. # TODO: add hashing of sensor data object to make them unique.
    """
    expanded_settings = []
    for s in settings:
        for i in range(repeats):
            new_settings = copy.copy(s)
            new_settings.rng_seed = i
            expanded_settings.append(new_settings)
    
    results_dir = os.path.join('data', results_dir)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    n_processed = 0
    to_process = []
    for s in expanded_settings:
        rel_path = os.path.join(results_dir, file_name(s, sensor_data))
        if not os.path.exists(rel_path):
            to_process.append((sensor_data, s))
        else:
            n_processed += 1

    print(f"Processing {len(to_process)} settings, {n_processed} already processed.")

    pool = mp.Pool(processes=pool_processes)
    results = pool.map(perform_slam, to_process)

    for s, res in zip(to_process, results):
        print(f"Saving results for {file_name(s[1], sensor_data)}")
        rel_path = os.path.join(results_dir, file_name(s[1], sensor_data))
        with open(rel_path, 'wb') as f:
            pickle.dump(res, f)

    res_ret = []
    for s in settings:
        res_settings_lst = []
        for i in range(repeats):
            new_settings = copy.copy(s)
            new_settings.rng_seed = i
            print(f"Loading results for {file_name(new_settings, sensor_data)}")
            rel_path = os.path.join(results_dir, file_name(new_settings, sensor_data))
            with open(rel_path, 'rb') as f:
                res_settings_lst.append(pickle.load(f))
        res_ret.append(res_settings_lst)

    return res_ret



if __name__ == "__main__":
    import numpy as np
    odom_mul_r_dtheta = [
        np.square(np.diag([r_noise, dtheta_noise])) 
        for r_noise, dtheta_noise in [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]]
    ]
    N = [10, 20, 50, 100, 200, 500]
    settings_collection = [
        fs.FastSLAMSettings(
            action_model_settings=ActionModelSettings(
                ODOM_MULT_COV=odom_cov
            ),
            num_particles=n,
        ) 
        for n in N[:1]
        for odom_cov in odom_mul_r_dtheta[:2]
    ]

    sensor_data = sd.load_sensor_data('sim3.xz')

    res = slam_batch(settings_collection, sensor_data, repeats=3, pool_processes=4)