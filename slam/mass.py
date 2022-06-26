from __future__ import annotations
import collections
import os
import pickle
import copy
import concurrent.futures
import time
import dataclasses
from typing import Callable

import numpy as np
from slam.action_model import ActionModelSettings

import slam.fastslam as fs
import slam.offline as offline
import sensor_data.sensor_data as sd
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
        ignore_existing=False,
        start_time=settings.t0,
        final_time=settings.tf
    )
    return res

def slam_batch(settings: list[fs.FastSLAMSettings], sensor_data: sd.SensorData, 
                repeats: int = 2, pool_processes: int = None, results_dir = 'slammed',
                stats_iter_size: int = 5, multiprocess:bool = False) -> list[fs.FastSLAM]:
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
    t0 = time.time()
    dt_iter = collections.deque([t0], maxlen=stats_iter_size)
    if len(to_process) > 0 and multiprocess:
        with concurrent.futures.ProcessPoolExecutor(max_workers=pool_processes) as executor:
            futures = [executor.submit(perform_slam, args_tuple) for args_tuple in to_process]
            for i, res_future in enumerate(concurrent.futures.as_completed(futures)):
                sensor_data, settings_inst = to_process[i]
                dt_iter.append(time.time() - dt_iter[-1])
                # print(f"Saving results for {file_name(s[1], sensor_data)}")
                print(f"{i+1:05d}/{len(to_process):05d} Jobs done. {time.time() - t0:.3f}s elapsed", end="\n")
                rel_path = os.path.join(results_dir, offline.file_name(settings_inst, sensor_data))
                
                res = res_future.result()
                if not os.path.exists(rel_path):
                    offline.save_slam_result(rel_path, res, settings_inst, sensor_data)
        if len(to_process):
            print(f"All jobs completed in {time.time() - t0:.3f} seconds. ({(time.time() - t0)/len(to_process):.3f}s per job)")
    else:
        for i, args in enumerate(to_process):
            sensor_data, settings_inst = args
            dt_iter.append(time.time() - dt_iter[-1])
            perform_slam(args)
            print(f"{i+1:05d}/{len(to_process):05d} Jobs done. {time.time() - t0:.3f}s elapsed", end="\n")
        if len(to_process):
            print(f"All jobs completed in {time.time() - t0:.3f} seconds. ({(time.time() - t0)/len(to_process):.3f}s per job)")
    print(f"Loading all results")    
    res_ret = []
    for s in settings:
        for i in range(repeats):
            new_settings = copy.copy(s)
            new_settings.rng_seed = i
            rel_path = os.path.join(results_dir, offline.file_name(new_settings, sensor_data))
            res = offline.load_slam_result(rel_path)
        res_ret.append(res)
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

def check_files(results_dir = 'slammed', sensor_data: sd.SensorData = None):
    results_dir = os.path.join('data', results_dir)
    ignore_dir = os.path.join('data', 'slammed_ignore')
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    if not os.path.isdir(ignore_dir):
        os.mkdir(ignore_dir)

    files = os.listdir(results_dir)
    files = [os.path.join(results_dir, file) for file in files if "." not in file] # ignore .txt, .png, etc (keep only data files)
    for idx, file in enumerate(files):
        name = os.path.basename(file)
        with open(file, 'rb') as f:
            data, settings_inst = pickle.load(f)
        if False: #'num_particles=11' in dif_repr(settings_inst):
            to_move = [files[idx], files[idx] +'.png', files[idx] +'.txt']
            new_names = [os.path.join(ignore_dir, os.path.basename(basename)) for basename in to_move]
            # print(f"{to_move=}")
            # print(f"{new_names=}")
            for file_to_move, new_name in zip(to_move, new_names):
                os.rename(file_to_move, new_name)
            print(f"{files[idx]} moved")
        if False and sensor_data is not None and name != offline.file_name(settings_inst, sensor_data):
            print("Skipping file for different sensor data")
            continue
        simulated = "[SIMULATED]" if data.actual_trajectory is not None else ""
        print(f"{files[idx]+'.png'} {simulated} -> {dif_repr(settings_inst)}")


def load_files_where(cond: Callable[[fs.SLAMResult, fs.FastSLAMSettings], bool], results_dir = 'slammed') -> list[tuple[fs.SLAMResult, fs.FastSLAMSettings]]:
    results_dir = os.path.join('data', results_dir)
    ignore_dir = os.path.join('data', 'slammed_ignore')
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    if not os.path.isdir(ignore_dir):
        os.mkdir(ignore_dir)

    files = os.listdir(results_dir)
    files = [os.path.join(results_dir, file) for file in files if "." not in file] # ignore .txt, .png, etc (keep only data files)
    results_tuple_to_return = []
    for idx, file in enumerate(files):
        name = os.path.basename(file)
        with open(file, 'rb') as f:
            data, settings_inst = pickle.load(f)
        if cond(data, settings_inst):
            results_tuple_to_return.append((data, settings_inst))
    return results_tuple_to_return

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
        type=int, 
        default=None,
        help="Maximum number of processes on which to run jobs. Default (None) uses the number of processors of the machine"
    )
    parser.add_argument(
        "--sequential", 
        action="store_true",
        help="Whether to sequentially perform the tasks"
    )
    parser.add_argument(
        "-N", 
        type=int, 
        nargs="+", 
        default=[def_set.num_particles]
    )
    parser.add_argument(
        "-t0", 
        type=float, 
        default=0
    )
    parser.add_argument(
        "-tf", 
        type=float, 
        default=np.inf
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
    
    sensor_data = sd.load_sensor_data(args.sensor_data)
    if args.check_files:
        check_files(sensor_data=sensor_data)
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
            t0=args.t0,
            tf=args.tf
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


    res = slam_batch(settings_collection, sensor_data, repeats=args.repeats, pool_processes=args.processes, multiprocess=not args.sequential)
