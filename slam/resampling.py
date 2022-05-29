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

class ResampleType(Enum):
    UNIFORM = resample_uniform
