import collections
import os
import pickle
import copy
import concurrent.futures
import time
import dataclasses

from slam.action_model import ActionModelSettings

import slam.fastslam as fs
import slam.offline as offline
import sensor_data.sensor_data as sd
import slam.plot_map as pm
import matplotlib.pyplot as plt

if not os.path.isdir('data'):
    os.mkdir('data')

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
        profile=False,
        ignore_existing=False
    )
    return res

def slam_batch(settings: list[fs.FastSLAMSettings], sensor_data: sd.SensorData, 
                repeats: int = 2, pool_processes: int = None, results_dir = 'slammed',
                stats_iter_size: int = 5) -> list[fs.FastSLAM]:
    """
    Run FastSLAM on a batch of settings, 'repeats' times per settings object.
    Returns a list of lists of fs.SLAMResult objects.
    Each list contains the results for one settings object for several seeds.

    These results are stored in files, indexed by the settings object hash.
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
        rel_path = os.path.join(results_dir, offline.file_name(s, sensor_data))
        if not os.path.exists(rel_path):
            to_process.append((sensor_data, s))
        else:
            n_processed += 1

    print(f"Processing {len(to_process)} settings, {n_processed} already processed.")
    if len(to_process) > 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=pool_processes) as executor:
            futures = [executor.submit(perform_slam, args_tuple) for args_tuple in to_process]
            t0 = time.time()
            dt_iter = collections.deque([t0], maxlen=stats_iter_size)
            for i, res_future in enumerate(concurrent.futures.as_completed(futures)):
                sensor_data, settings_inst = to_process[i]
                dt_iter.append(time.time() - dt_iter[-1])
                # print(f"Saving results for {file_name(s[1], sensor_data)}")
                print(f"{i+1:05d}/{len(to_process):05d} Jobs done. {time.time() - t0:.3f}s elapsed", end="\n")
                rel_path = os.path.join(results_dir, offline.file_name(settings_inst, sensor_data))
                
                res = res_future.result()
                if not os.path.exists(rel_path):
                    offline.save_slam_result(rel_path, res, settings_inst, sensor_data)
        print(f"All jobs completed in {time.time() - t0:.3f} seconds. ({(time.time() - t0)/len(to_process):.3f} per job)")
    print(f"Loading all results")    
    res_ret = []
    for s in settings:
        res_settings_lst = []
        for i in range(repeats):
            new_settings = copy.copy(s)
            new_settings.rng_seed = i
            rel_path = os.path.join(results_dir, offline.file_name(new_settings, sensor_data))
            res = offline.load_slam_result(rel_path)
        res_ret.append(res_settings_lst)
    return res_ret

def flatten_dict(dict1):
    for key in list(dict1.keys()):
        if isinstance(dict1[key], dict):
            flatten_dict(dict1[key])
            items = list(dict1[key].items())
            dict1.pop(key)
            for key, val in items:
                dict1[key] = val
    return dict1

def dif_repr(settings_inst: fs.FastSLAMSettings()):
    def_set_dict = flatten_dict(dataclasses.asdict(fs.FastSLAMSettings()))
    settings_inst_dict = flatten_dict(dataclasses.asdict(settings_inst))
    diff = []
    for setting in settings_inst_dict:
        if setting in ["visualize", "trajectory_trail", "map_type"]:
            continue
        if isinstance(settings_inst_dict[setting], np.ndarray):
            if np.all(settings_inst_dict[setting] != def_set_dict[setting]):
                diff.append(f"{setting}={settings_inst_dict[setting]}")
        else:
            if settings_inst_dict[setting] != def_set_dict[setting]:
                diff.append(f"{setting}={settings_inst_dict[setting]}")
    diff.sort()
    if not diff:
        return "default"
    return (" ".join(diff)).replace("\n", "")

def check_files(results_dir = 'slammed'):
    results_dir = os.path.join('data', results_dir)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    def argsort(lst):
        return sorted(range(len(lst)), key=lst.__getitem__)

    files = os.listdir(results_dir)
    files = [os.path.join(results_dir, file) for file in files if "." not in file] # ignore .txt, .png, etc (keep only data files)
    characteristics = []
    for file in files:
        with open(file, 'rb') as f:
            data, settings_inst = pickle.load(f)
            characteristics.append(dif_repr(settings_inst))

    for idx in argsort(characteristics):
        print(f"{files[idx]} -> {characteristics[idx]}")


if __name__ == "__main__":
    import numpy as np
    import argparse

    # examples
    # python -m slam.mass --sensor-data simx.xz -N 1 2 3 
    # python -m slam-mass --check-files

    def_set = fs.FastSLAMSettings()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-files", action='store_true'
    )
    parser.add_argument(
        "--repeats", 
        type=int, 
        default=3,
        help="Number of times to repeat the SLAM algorithm for a given context (settings and SensorData)"
    )
    parser.add_argument(
        "--sensor-data", 
        type=str, 
        default="sim0.xz",
        help="Name of file for SensorData object to apply SLAM onto"
    )
    parser.add_argument(
        "--processes", 
        default=None,
        help="Maximum number of processes on which to run jobs. Default (None) uses the number of processors of the machine"
    )
    parser.add_argument(
        "-N", 
        type=int, 
        nargs="+", 
        default=[def_set.num_particles]
    )
    parser.add_argument(
        "--action-model-noise-cov", 
        type=str, 
        default=f"[[{def_set.action_model_settings.ODOM_MULT_COV[0, 0]}, {def_set.action_model_settings.ODOM_MULT_COV[1, 1]}]]"
    )
    parser.add_argument(
        "--action-model-noise-bias", 
        type=str, 
        default=f"[[{def_set.action_model_settings.ODOM_MULT_MU[0]}, {def_set.action_model_settings.ODOM_MULT_MU[1]}]]"
    )
    parser.add_argument(
        "-r-std", 
        type=float, 
        nargs="+", 
        default=[def_set.r_std]
    )
    parser.add_argument(
        "-phi-std", 
        type=float, 
        nargs="+", 
        default=[def_set.phi_std]
    )
    parser.add_argument(
        "-psi-std", 
        type=float, 
        nargs="+", 
        default=[def_set.psi_std]
    )
    parser.add_argument(
        "-r-std-line", 
        type=float, 
        nargs="+", 
        default=[def_set.r_std_line]
    )
    parser.add_argument(
        "-phi-std-line", 
        type=float, 
        nargs="+", 
        default=[def_set.phi_std_line]
    )
    args = parser.parse_args()
    if args.check_files:
        check_files()
        exit(0)

    odom_mul_r_dtheta_cov = [
        np.square(np.diag([r_noise, dtheta_noise])) 
        for r_noise, dtheta_noise in eval(args.action_model_noise_cov)
    ]
    odom_mul_r_dtheta_mu = [
        np.array([r_noise, dtheta_noise])
        for r_noise, dtheta_noise in eval(args.action_model_noise_bias)
    ]

    settings_collection = [
        fs.FastSLAMSettings(
            action_model_settings=ActionModelSettings(
                ODOM_MULT_COV=odom_cov,
                ODOM_MULT_MU=odom_mu
            ),
            num_particles=n,
            r_std=r_std,
            phi_std=phi_std,
            psi_std=psi_std,
            r_std_line=r_std_line,
            phi_std_line=phi_std_line,
        ) 
        for n in args.N
        for odom_cov in odom_mul_r_dtheta_cov
        for odom_mu in odom_mul_r_dtheta_mu
        for r_std in args.r_std
        for phi_std in args.phi_std
        for psi_std in args.psi_std
        for r_std_line in args.r_std_line
        for phi_std_line in args.phi_std_line
    ]

    sensor_data = sd.load_sensor_data(args.sensor_data)

    res = slam_batch(settings_collection, sensor_data, repeats=args.repeats, pool_processes=args.processes)
