import multiprocessing as mp
import os
import pickle
import copy

import slam.fastslam as fs
import slam.offline as offline
import sensor_data.sensor_data as sd

if not os.path.isdir('data'):
    os.mkdir('data')

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
        rel_path = os.path.join(results_dir, file_name(s))
        if not os.path.exists(rel_path):
            to_process.append((sensor_data, s))
        else:
            n_processed += 1

    print(f"Processing {len(to_process)} settings, {n_processed} already processed.")

    pool = mp.Pool(processes=pool_processes)
    results = pool.map(perform_slam, to_process)

    for s, res in zip(to_process, results):
        print(f"Saving results for {file_name(s[1])}")
        rel_path = os.path.join(results_dir, file_name(s[1]))
        with open(rel_path, 'wb') as f:
            pickle.dump(res, f)

    res_ret = []
    for s in settings:
        res_settings_lst = []
        for i in range(repeats):
            print(f"Loading results for {file_name(s)}")
            new_settings = copy.copy(s)
            new_settings.rng_seed = i
            rel_path = os.path.join(results_dir, file_name(new_settings))
            with open(rel_path, 'rb') as f:
                res_settings_lst.append(pickle.load(f))
        res_ret.append(res_settings_lst)

    return res_ret



if __name__ == "__main__":
    settings_collection = [
        fs.FastSLAMSettings(
            num_particles=n,
        ) for n in [5, 10]
    ]

    sensor_data = sd.load_sensor_data('sim5.xz')

    res = slam_batch(settings_collection, sensor_data, repeats=3, pool_processes=4)