"""
Player class for managing individual player state and tracking history.
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple, Optional
from .utils import (
    bbox_area, init_kalman_filter, estimate_bbox_from_center,
    clamp_bbox_to_frame
)


class Player:
    """
    Represents a single player with tracking state, appearance features,
    and trajectory history.
    """

    def __init__(self, player_id: int, bbox: Tuple[int, int, int, int],
                 center: Tuple[int, int], frame_idx: int,
                 shirt_hist: Optional[np.ndarray] = None,
                 pants_hist: Optional[np.ndarray] = None,
                 color_category: str = 'other',
                 color_values: Optional[Tuple] = None,
                 config=None):
        """
        Initialize a player.

        Args:
            player_id: Unique player identifier
            bbox: Initial bounding box (x1, y1, x2, y2)
            center: Initial center position (cx, cy)
            frame_idx: Frame index when player was initialized
            shirt_hist: Initial shirt histogram
            pants_hist: Initial pants histogram
            color_category: Color category string
            color_values: Color HSV values
            config: TrackingConfig instance
        """
        self.id = player_id
        self.last_bbox = bbox
        self.last_center = center

        # Bbox dimensions
        x1, y1, x2, y2 = bbox
        self.avg_width = x2 - x1
        self.avg_height = y2 - y1

        # Initialize Kalman filter
        if config is not None:
            self.kalman = init_kalman_filter(
                center[0], center[1],
                config.kalman_process_noise,
                config.kalman_measurement_noise
            )
        else:
            self.kalman = init_kalman_filter(center[0], center[1])

        # Trajectory and history
        traj_len = config.trajectory_max_len if config else 300
        pos_len = config.position_history_len if config else 10
        size_len = config.bbox_size_history_len if config else 20

        self.trajectory = deque(maxlen=traj_len)
        self.bird_trajectory = deque(maxlen=traj_len)
        self.position_history = deque(maxlen=pos_len)
        self.bbox_size_history = deque(maxlen=size_len)

        # Initialize history
        initial_area = bbox_area(bbox)
        self.position_history.append(center)
        self.bbox_size_history.append(initial_area)
        self.avg_bbox_size = initial_area

        # Distance tracking
        self.total_distance = 0.0
        self.last_bird_pos = None

        # Appearance features
        self.shirt_hist = shirt_hist
        self.pants_hist = pants_hist
        self.shirt_hist_original = shirt_hist.copy() if shirt_hist is not None else None
        self.pants_hist_original = pants_hist.copy() if pants_hist is not None else None
        self.color_category = color_category
        self.color_category_original = color_category
        self.color_values = color_values

        # Tracking state
        self.found_frames = 0
        self.stability = 5.0
        self.prediction_frames = 0  # How many frames in prediction mode
        self.consecutive_matches = 0
        self.last_seen_frame = frame_idx

        # Visualization
        self.color = None  # Will be set by tracker

    def predict(self):
        """Predict next position using Kalman filter."""
        self.kalman.predict()

    def get_predicted_center(self) -> Tuple[float, float]:
        """
        Get predicted center from Kalman filter.

        Returns:
            (pred_x, pred_y) predicted center
        """
        state = self.kalman.statePost
        return float(state[0, 0]), float(state[1, 0])

    def update_with_detection(self, bbox: Tuple[int, int, int, int],
                              center: Tuple[int, int],
                              frame_idx: int,
                              shirt_hist: Optional[np.ndarray] = None,
                              pants_hist: Optional[np.ndarray] = None,
                              hist_alpha: float = 0.95):
        """
        Update player state with new detection.

        Args:
            bbox: New bounding box
            center: New center position
            frame_idx: Current frame index
            shirt_hist: New shirt histogram
            pants_hist: New pants histogram
            hist_alpha: Histogram smoothing factor
        """
        from .appearance import AppearanceModel

        x1, y1, x2, y2 = bbox
        cx, cy = center
        w = x2 - x1
        h = y2 - y1

        # Update Kalman filter
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        corrected = self.kalman.correct(measurement)

        fx = int(corrected[0, 0])
        fy = int(corrected[1, 0])

        # Update state
        self.last_bbox = bbox
        self.last_center = (fx, fy)
        self.avg_width = int(self.avg_width * 0.9 + w * 0.1)
        self.avg_height = int(self.avg_height * 0.9 + h * 0.1)

        # Update trajectory
        self.trajectory.append((fx, fy))
        self.position_history.append((cx, cy))

        # Update bbox size
        current_area = bbox_area(bbox)
        self.bbox_size_history.append(current_area)
        self.avg_bbox_size = float(np.mean(self.bbox_size_history))

        # Update tracking state
        self.found_frames += 1
        self.stability = min(self.stability + 0.3, 10.0)
        self.prediction_frames = 0
        self.consecutive_matches += 1
        self.last_seen_frame = frame_idx

        # Update histograms (only if stable tracking)
        if self.consecutive_matches > 4:
            if shirt_hist is not None:
                self.shirt_hist = AppearanceModel.smooth_histogram(
                    self.shirt_hist, shirt_hist, hist_alpha
                )
            if pants_hist is not None:
                self.pants_hist = AppearanceModel.smooth_histogram(
                    self.pants_hist, pants_hist, hist_alpha
                )

    def update_with_prediction(self, frame_width: int, frame_height: int,
                              max_prediction_frames: int = 60):
        """
        Update player state using Kalman prediction (no detection).

        Args:
            frame_width: Frame width for clamping
            frame_height: Frame height for clamping
            max_prediction_frames: Maximum frames to predict
        """
        # Get predicted position
        pred_x, pred_y = self.get_predicted_center()
        pred_x = int(pred_x)
        pred_y = int(pred_y)

        # Update state
        self.stability = max(self.stability - 0.15, 0.0)
        self.prediction_frames += 1

        if self.prediction_frames > max_prediction_frames:
            self.prediction_frames = max_prediction_frames
            self.stability = 0.0

        self.consecutive_matches = 0

        # Update position
        self.last_center = (pred_x, pred_y)
        self.trajectory.append((pred_x, pred_y))
        self.position_history.append((pred_x, pred_y))

        # Estimate bbox from predicted center
        est_bbox = estimate_bbox_from_center(
            (pred_x, pred_y),
            self.avg_width,
            self.avg_height
        )
        est_bbox = clamp_bbox_to_frame(est_bbox, frame_width, frame_height)
        self.last_bbox = est_bbox

        # Reset appearance if lost for too long
        if self.prediction_frames > 30:
            if self.shirt_hist_original is not None:
                self.shirt_hist = self.shirt_hist_original.copy()
            if self.pants_hist_original is not None:
                self.pants_hist = self.pants_hist_original.copy()
            self.color_category = self.color_category_original

    def update_bird_view_position(self, bird_pos: Tuple[int, int],
                                  pixels_per_meter_x: float,
                                  pixels_per_meter_y: float):
        """
        Update bird's eye view position and calculate distance traveled.

        Args:
            bird_pos: (bird_x, bird_y) position in bird view
            pixels_per_meter_x: Conversion factor X
            pixels_per_meter_y: Conversion factor Y
        """
        self.bird_trajectory.append(bird_pos)

        # Calculate distance (only for active tracking, not prediction)
        if self.last_bird_pos is not None and self.prediction_frames <= 2:
            dx_m = (bird_pos[0] - self.last_bird_pos[0]) / pixels_per_meter_x
            dy_m = (bird_pos[1] - self.last_bird_pos[1]) / pixels_per_meter_y
            dist = np.sqrt(dx_m**2 + dy_m**2)

            # Sanity check: ignore unrealistic jumps
            if dist < 3.0:
                self.total_distance += dist

        self.last_bird_pos = bird_pos

    def is_locked(self, lock_threshold: int = 6) -> bool:
        """
        Check if player tracking is locked (stable).

        Args:
            lock_threshold: Consecutive matches threshold

        Returns:
            True if locked
        """
        return self.consecutive_matches >= lock_threshold

    def is_active(self) -> bool:
        """
        Check if player is actively tracked (not in prediction mode).

        Returns:
            True if actively matched
        """
        return self.prediction_frames == 0

    def get_display_color(self) -> Tuple[int, int, int]:
        """
        Get display color for visualization.

        Returns:
            BGR color tuple
        """
        if self.color is not None:
            return self.color

        # Generate color from ID if not set
        hue = (self.id * 47) % 180
        color = tuple(map(int, cv2.cvtColor(
            np.uint8([[[hue, 255, 255]]]),
            cv2.COLOR_HSV2BGR
        )[0][0]))
        return color

    def get_label(self) -> str:
        """
        Get display label for visualization.

        Returns:
            Label string
        """
        if self.is_active():
            return f"ID {self.id} (S:{self.stability:.1f})"
        else:
            return f"ID {self.id} [P:{self.prediction_frames}]"

    def __repr__(self) -> str:
        return (f"Player(id={self.id}, center={self.last_center}, "
                f"active={self.is_active()}, stability={self.stability:.1f})")
