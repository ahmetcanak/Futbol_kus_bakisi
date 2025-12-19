"""
Configuration module for multi-player tracking system.
Contains all hyperparameters and settings.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TrackingConfig:
    """Main configuration for the tracking system."""

    # Video paths
    video_path: str = "shs2.mp4"
    output_path: str = "shs2_fixed_players_v5.mp4"
    yolo_model: str = "yolov8m.pt"

    # Player detection
    fixed_player_count: int = 14
    max_init_frames: int = 60
    yolo_conf_threshold: float = 0.25
    yolo_conf_init: float = 0.20  # Lower confidence for initialization
    yolo_iou_threshold: float = 0.4
    yolo_imgsz: int = 1280

    # Re-identification parameters
    max_reid_dist: float = 35**2
    max_reid_dist_predicted: float = 60**2
    max_prediction_frames: int = 60

    # Histogram parameters
    hist_size: Tuple[int, int] = (32, 32)
    hist_range: List[float] = field(default_factory=lambda: [0, 180, 0, 256])
    shirt_weight: float = 18.0
    pants_weight: float = 12.0
    hist_alpha: float = 0.95  # Smoothing factor

    # Anti-swap parameters
    velocity_weight: float = 8.0
    min_iou_for_match: float = 0.15
    max_velocity_change: float = 45
    distance_penalty: float = 0.10
    direction_weight: float = 6.0
    size_weight: float = 10.0
    max_size_change: float = 0.40

    # Spatial control
    max_x_jump: float = 120
    max_y_jump: float = 80
    spatial_penalty: float = 12.0

    # Stability parameters
    position_history_len: int = 10
    lock_threshold: int = 6
    trajectory_max_len: int = 300
    bbox_size_history_len: int = 20

    # Kalman filter parameters
    kalman_process_noise: float = 0.03
    kalman_measurement_noise: float = 2.0

    # Bird's eye view configuration
    bird_width: int = 500
    bird_height: int = 600
    field_width_m: float = 15.0
    field_height_m: float = 30.0

    # Perspective transform points
    src_points: np.ndarray = field(default_factory=lambda: np.float32([
        [115, 370],   # tl
        [250, 1070],  # bl
        [680, 310],   # tr
        [1910, 460],  # br
        [1600, 1070], # bb
        [1150, 345],  # tbr
    ]))

    dst_points: np.ndarray = field(default_factory=lambda: np.float32([
        [50, 50],
        [50, 550],
        [350, 50],
        [350, 550],
        [200, 550],
        [350, 300],
    ]))

    @property
    def pixels_per_meter_x(self) -> float:
        """Calculate pixels per meter in X direction."""
        return 300.0 / self.field_width_m

    @property
    def pixels_per_meter_y(self) -> float:
        """Calculate pixels per meter in Y direction."""
        return 500.0 / self.field_height_m

    def get_homography_matrix(self) -> np.ndarray:
        """Compute homography matrix for bird's eye view transformation."""
        import cv2
        matrix, _ = cv2.findHomography(
            self.src_points,
            self.dst_points,
            method=cv2.RANSAC
        )
        return matrix


class ColorCategory:
    """Color category definitions."""
    BLACK: str = 'black'
    YELLOW: str = 'yellow'
    YELLOW_STRIPE: str = 'yellow_stripe'
    GREEN_VEST: str = 'green_vest'
    WHITE: str = 'white'
    OTHER: str = 'other'

    SIMILAR_PAIRS: List[Tuple[str, str]] = [
        ('yellow', 'yellow_stripe'),
        ('yellow', 'green_vest'),
    ]

    @classmethod
    def get_similarity_penalty(cls, cat1: str, cat2: str) -> float:
        """Get penalty for color category mismatch."""
        if cat1 == cat2:
            return 0.0

        for pair in cls.SIMILAR_PAIRS:
            if cat1 in pair and cat2 in pair:
                return 5.0

        return 20.0
