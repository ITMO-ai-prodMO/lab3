"""Filtering algorithms implemented for the lab."""

from .kalman import kalman_random_walk
from .moving_average import moving_average
from .savitzky_golay import savitzky_golay
from .wavelet import haar_wavelet_denoise

__all__ = ["haar_wavelet_denoise", "kalman_random_walk", "moving_average", "savitzky_golay"]
