"""
Multi-player tracking system with fixed IDs and anti-swap logic.

This package provides a modular, class-based implementation for tracking
multiple football players in video with stable ID assignment.
"""

from .config import TrackingConfig, ColorCategory
from .player import Player
from .appearance import AppearanceModel
from .tracker import MultiPlayerTracker
from .video_processor import VideoProcessor
from . import utils

__version__ = "2.0.0"
__all__ = [
    "TrackingConfig",
    "ColorCategory",
    "Player",
    "AppearanceModel",
    "MultiPlayerTracker",
    "VideoProcessor",
    "utils",
]
