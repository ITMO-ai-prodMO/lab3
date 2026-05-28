"""Filtering algorithms implemented for the lab."""

from .kalman import kalman_random_walk
from .moving_average import moving_average
from .savitzky_golay import savitzky_golay

__all__ = ["kalman_random_walk", "moving_average", "savitzky_golay"]
