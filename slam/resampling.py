from __future__ import annotations
import numpy as np

from slam.particle import Particle

from enum import Enum


def get_index(r, probabilities):
    """ 
    Returns the index from probabilities that corresponds to r.
    r is in [0, 1[, that is, 0 is included in the half-open interval and 1 is not. 
    Calling with uniform random r gives a random index with the probabilities of probabilities.
    Probabilities must be non-negative and sum to 1.
    
    Example: 
        probabilities = [0, 1/4, 0, 1/2, 1/4, 0 ]

        r in [0, 0.25[  ->  index: 1
        r in [0.25, 0.75[  ->  index: 3
        r in [0.75, 1[      -> index: 4
    """
    i = 0
    while probabilities[i] <= r:
        r -= probabilities[i]
        i += 1
    return min(i, len(probabilities) - 1)


def resample_uniform(particles: list[Particle], weights: np.ndarray, N: int) -> np.ndarray:
    """
    Resample N particles according to a uniform distribution.

    Args:
        weights: The weights of the particles.
        N: number of particles to sample.

    Returns:
        The resampled particles.
    """
    new_particles = []
    new_weight = 1/N
    probs = weights/np.sum(weights)
    for _ in range(N):
        i = get_index(np.random.uniform(), probs)
        new_particle = particles[i].copy()
        new_particle.weight = new_weight
        new_particles.append(new_particle)
    return new_particles


def resample_low_variance(particles: list[Particle], weights: np.ndarray, N: int) -> np.ndarray:
    """
    Resample N particles using low variance resampling.

    Args:
        weights: The weights of the particles.
        N: number of particles to sample.

    Returns:
        The resampled particles.
    """
    start = np.random.random()
    step = 1.0 / N
    new_particles = []
    new_weight = 1/N
    probs = weights/np.sum(weights)
    for k in range(N):
        i = get_index((start + k*step) % 1, probs)
        new_particle = particles[i].copy()
        new_particle.weight = new_weight
        new_particles.append(new_particle)

    return new_particles

class ResampleType(Enum):
    UNIFORM = resample_uniform
    LOW_VARIANCE = resample_low_variance


if __name__ == "__main__":
    particles = [Particle(pose=i) for i in range(5)]
    weights = np.array([0.2, 3, 1, 1, 0])
    tries = 5
    uni = []
    lv = []
    print(f"id's before resample {[p.pose.item() for p in particles]}")
    for i in range(tries):
        new_uni = resample_uniform(particles, weights, 5)
        uni.append(sorted([p.pose.item() for p in new_uni]))
        new_lv = resample_low_variance(particles, weights, 5)
        lv.append(sorted([p.pose.item() for p in new_lv]))

    print(f"with uniform sampling: {uni}")
    print(f"with low variance sampling {lv}")
    print("Each sub-list contains the result of a resample.")
