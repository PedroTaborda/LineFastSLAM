from __future__ import annotations
import numpy as np

from slam.particle import Particle

from enum import Enum

def resample_uniform(particles: list[Particle], weights: np.ndarray) -> np.ndarray:
    """
    Resample particles according to a uniform distribution.

    Args:
        weights: The weights of the particles.

    Returns:
        The indices of the resampled particles.
    """
    idxs = np.random.choice(np.arange(len(weights)), size=len(weights), p=weights/np.sum(weights))
    new_particles = [particles[i].copy() for i in idxs]
    for particle in new_particles:
        particle.weight = 1.0# / len(new_particles)
    return new_particles

def resample_low_variance(particles: list[Particle], weights: np.ndarray) -> np.ndarray:
    """
    Resample particles using low variance resampling.

    Args:
        weights: The weights of the particles.

    Returns:
        The indices of the resampled particles.
    """
    start = np.random.random()
    step = 1.0 / len(weights)
    weights = weights / np.sum(weights)
    new_particles = []

    weights_sum = 0
    # place start in the correct bin
    i_start = 0
    weights_sum += weights[i_start]
    while weights[i_start] < start:
        i_start += 1

    i = i_start
    w = (start + i * step) % 1
    while ((i - i_start) % len(weights)) * step < 1:
        w = (start + i * step) % 1
        while weights[i] <= w:
            i += 1
            i %= len(weights)
        new_particles.append(particles[i].copy())

    for particle in new_particles:
        particle.weight = 1.0 / len(new_particles)

    return new_particles

class ResampleType(Enum):
    UNIFORM = resample_uniform
    LOW_VARIANCE = resample_low_variance
