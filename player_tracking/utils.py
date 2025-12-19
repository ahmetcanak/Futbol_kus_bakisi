"""
Utility functions for bounding box operations, transformations, and calculations.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def bbox_iou(boxA: Tuple[int, int, int, int],
             boxB: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA: (x1, y1, x2, y2) coordinates
        boxB: (x1, y1, x2, y2) coordinates

    Returns:
        IoU value between 0 and 1
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0

    return interArea / union


def bbox_area(bbox: Tuple[int, int, int, int]) -> float:
    """
    Calculate area of a bounding box.

    Args:
        bbox: (x1, y1, x2, y2) coordinates

    Returns:
        Area in pixels
    """
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def clamp_bbox_to_frame(bbox: Tuple[int, int, int, int],
                        width: int,
                        height: int) -> Tuple[int, int, int, int]:
    """
    Clamp bounding box coordinates to frame boundaries.

    Args:
        bbox: (x1, y1, x2, y2) coordinates
        width: Frame width
        height: Frame height

    Returns:
        Clamped bbox coordinates
    """
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    # Ensure valid bbox
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)

    return (x1, y1, x2, y2)


def estimate_bbox_from_center(center: Tuple[int, int],
                               avg_width: int,
                               avg_height: int) -> Tuple[int, int, int, int]:
    """
    Estimate bounding box from center point and average dimensions.

    Args:
        center: (cx, cy) center coordinates
        avg_width: Average width
        avg_height: Average height

    Returns:
        Estimated bbox (x1, y1, x2, y2)
    """
    cx, cy = center
    half_w = max(1, avg_width // 2)
    half_h = max(1, avg_height // 2)

    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def compute_spatial_jump(last_center: Tuple[int, int],
                        new_center: Tuple[int, int]) -> Tuple[float, float]:
    """
    Compute spatial jump between two center points.

    Args:
        last_center: (x, y) previous center
        new_center: (x, y) current center

    Returns:
        (dx, dy) absolute differences
    """
    dx = abs(new_center[0] - last_center[0])
    dy = abs(new_center[1] - last_center[1])
    return dx, dy


def compute_shirt_pants_bboxes(bbox: Tuple[int, int, int, int]) -> Tuple[
    Optional[Tuple[int, int, int, int]],
    Optional[Tuple[int, int, int, int]]
]:
    """
    Compute separate bounding boxes for shirt and pants regions.

    Args:
        bbox: Full person bbox (x1, y1, x2, y2)

    Returns:
        (shirt_bbox, pants_bbox) or (None, None) if invalid
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if h <= 0 or w <= 0:
        return None, None

    # Add horizontal margin to avoid arms
    margin_w = int(w * 0.25)
    inner_x1 = x1 + margin_w
    inner_x2 = x2 - margin_w

    if inner_x2 <= inner_x1:
        inner_x1, inner_x2 = x1, x2

    # Shirt region: 18-48% of height
    shirt_top = y1 + int(h * 0.18)
    shirt_bottom = y1 + int(h * 0.48)

    # Pants region: 48-78% of height
    pants_top = y1 + int(h * 0.48)
    pants_bottom = y1 + int(h * 0.78)

    shirt_bbox = (inner_x1, max(y1, shirt_top), inner_x2, min(y2, shirt_bottom))
    pants_bbox = (inner_x1, max(y1, pants_top), inner_x2, min(y2, pants_bottom))

    return shirt_bbox, pants_bbox


def init_kalman_filter(x: float, y: float,
                       process_noise: float = 0.03,
                       measurement_noise: float = 2.0) -> cv2.KalmanFilter:
    """
    Initialize Kalman filter for position tracking.

    Args:
        x: Initial x position
        y: Initial y position
        process_noise: Process noise covariance
        measurement_noise: Measurement noise covariance

    Returns:
        Initialized Kalman filter
    """
    kf = cv2.KalmanFilter(4, 2)

    # State transition matrix [x, y, vx, vy]
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], np.float32)

    # Measurement matrix (we only measure position)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
    kf.statePost = np.array([[x], [y], [0.0], [0.0]], np.float32)

    return kf


def compute_velocity_consistency(kalman: cv2.KalmanFilter,
                                 new_cx: float,
                                 new_cy: float) -> float:
    """
    Compute velocity consistency between predicted and actual movement.

    Args:
        kalman: Kalman filter with current state
        new_cx: New center x coordinate
        new_cy: New center y coordinate

    Returns:
        Velocity difference magnitude
    """
    state = kalman.statePost
    pred_vx = state[2, 0]
    pred_vy = state[3, 0]

    prev_x = state[0, 0]
    prev_y = state[1, 0]

    actual_vx = new_cx - prev_x
    actual_vy = new_cy - prev_y

    vx_diff = abs(actual_vx - pred_vx)
    vy_diff = abs(actual_vy - pred_vy)

    return np.sqrt(vx_diff**2 + vy_diff**2)


def compute_direction_consistency(position_history,
                                  new_cx: float,
                                  new_cy: float) -> float:
    """
    Compute direction consistency based on position history.

    Args:
        position_history: Deque of recent positions
        new_cx: New center x coordinate
        new_cy: New center y coordinate

    Returns:
        Direction penalty (0 = consistent, higher = inconsistent)
    """
    if len(position_history) < 3:
        return 0.0

    recent_positions = list(position_history)[-5:]

    if len(recent_positions) < 2:
        return 0.0

    # Calculate average direction from history
    dx_avg = 0
    dy_avg = 0
    count = 0

    for i in range(1, len(recent_positions)):
        dx_avg += recent_positions[i][0] - recent_positions[i-1][0]
        dy_avg += recent_positions[i][1] - recent_positions[i-1][1]
        count += 1

    if count == 0:
        return 0.0

    dx_avg /= count
    dy_avg /= count

    # Calculate new direction
    last_pos = recent_positions[-1]
    new_dx = new_cx - last_pos[0]
    new_dy = new_cy - last_pos[1]

    mag_avg = np.sqrt(dx_avg**2 + dy_avg**2)
    mag_new = np.sqrt(new_dx**2 + new_dy**2)

    # If movement is too small, don't penalize
    if mag_avg < 3 or mag_new < 3:
        return 0.0

    # Normalize vectors
    dx_avg /= mag_avg
    dy_avg /= mag_avg
    new_dx /= mag_new
    new_dy /= mag_new

    # Compute dot product (cosine similarity)
    dot = dx_avg * new_dx + dy_avg * new_dy
    direction_penalty = max(0, 1 - dot)

    return direction_penalty


def compute_size_consistency(last_bbox: Tuple[int, int, int, int],
                             new_bbox: Tuple[int, int, int, int],
                             avg_size: float,
                             max_change: float = 0.40) -> float:
    """
    Compute size consistency between bboxes.

    Args:
        last_bbox: Previous bounding box
        new_bbox: New bounding box
        avg_size: Average bbox size
        max_change: Maximum allowed size change ratio

    Returns:
        Size inconsistency penalty
    """
    last_area = bbox_area(last_bbox)
    new_area = bbox_area(new_bbox)

    if last_area == 0 or new_area == 0:
        return 1.0

    # Compare with last bbox
    ratio1 = min(last_area, new_area) / max(last_area, new_area)

    # Compare with average size
    if avg_size > 0:
        ratio2 = min(avg_size, new_area) / max(avg_size, new_area)
    else:
        ratio2 = ratio1

    avg_ratio = (ratio1 + ratio2) / 2

    # High penalty if size change exceeds threshold
    if avg_ratio < (1 - max_change):
        return (1 - avg_ratio) * 2

    return (1 - avg_ratio)


def transform_point_to_bird_view(point: Tuple[int, int],
                                  matrix: np.ndarray) -> Tuple[int, int]:
    """
    Transform a point from camera view to bird's eye view.

    Args:
        point: (x, y) coordinates in camera view
        matrix: Homography matrix

    Returns:
        (bird_x, bird_y) coordinates in bird's eye view
    """
    pt = np.array([[point[0], point[1]]], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pt, matrix)
    bird_x = int(transformed[0, 0, 0])
    bird_y = int(transformed[0, 0, 1])
    return bird_x, bird_y
