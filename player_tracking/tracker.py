"""
Multi-player tracker with Hungarian algorithm matching and anti-swap logic.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional
from .player import Player
from .appearance import AppearanceModel
from .config import TrackingConfig, ColorCategory
from .utils import (
    bbox_iou, compute_spatial_jump, compute_velocity_consistency,
    compute_direction_consistency, compute_size_consistency
)


class MultiPlayerTracker:
    """
    Manages tracking of multiple players with fixed IDs and anti-swap logic.
    Uses Hungarian algorithm for optimal assignment.
    """

    def __init__(self, config: TrackingConfig):
        """
        Initialize tracker.

        Args:
            config: Tracking configuration
        """
        self.config = config
        self.players: Dict[int, Player] = {}
        self.appearance_model = AppearanceModel(
            hist_size=config.hist_size,
            hist_range=config.hist_range
        )

    def initialize_players(self, frame: np.ndarray,
                          bboxes: np.ndarray,
                          frame_idx: int) -> int:
        """
        Initialize fixed number of players from first frame detections.

        Args:
            frame: First frame
            bboxes: Detected bounding boxes
            frame_idx: Frame index

        Returns:
            Number of initialized players
        """
        for player_id in range(len(bboxes)):
            box = bboxes[player_id]
            x1, y1, x2, y2 = map(int, box)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Extract appearance features
            shirt_hist, pants_hist = self.appearance_model.get_shirt_pants_hist(
                frame, (x1, y1, x2, y2)
            )
            color_cat, color_vals = self.appearance_model.get_color_category(
                frame, (x1, y1, x2, y2)
            )

            # Create player
            player = Player(
                player_id=player_id,
                bbox=(x1, y1, x2, y2),
                center=center,
                frame_idx=frame_idx,
                shirt_hist=shirt_hist,
                pants_hist=pants_hist,
                color_category=color_cat,
                color_values=color_vals,
                config=self.config
            )

            # Assign visualization color
            hue = (player_id * 47) % 180
            import cv2
            color = tuple(map(int, cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]),
                cv2.COLOR_HSV2BGR
            )[0][0]))
            player.color = color

            self.players[player_id] = player

            print(f"  Player {player_id}: {color_cat}, "
                  f"pos=({center[0]}, {center[1]}), "
                  f"size={x2-x1}x{y2-y1}")

        return len(self.players)

    def predict_all(self):
        """Predict next positions for all players using Kalman filters."""
        for player in self.players.values():
            player.predict()

    def build_cost_matrix(self, detections: List[dict],
                         frame_width: int,
                         frame_height: int) -> Tuple[
        Optional[np.ndarray], List[int], List[dict]
    ]:
        """
        Build cost matrix for Hungarian algorithm.

        Args:
            detections: List of detection dictionaries
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            (cost_matrix, player_ids, detections) or (None, ids, []) if no detections
        """
        n_players = len(self.players)
        n_dets = len(detections)

        if n_players == 0:
            return None, [], []

        player_ids = list(self.players.keys())

        # If no detections, all players go to prediction mode
        if n_dets == 0:
            return None, player_ids, []

        # Initialize cost matrix with high values
        cost_matrix = np.full((n_players, n_dets), 1e6, dtype=np.float32)

        # Fill cost matrix
        for pi, player_id in enumerate(player_ids):
            player = self.players[player_id]

            # Get player state
            pred_x, pred_y = player.get_predicted_center()
            last_bbox = player.last_bbox
            last_center = player.last_center
            is_locked = player.is_locked(self.config.lock_threshold)
            prediction_frames = player.prediction_frames

            # Adaptive thresholds based on player state
            if prediction_frames > 0:
                # In prediction mode: more lenient
                max_dist = self.config.max_reid_dist_predicted
                max_x = self.config.max_x_jump * (1 + prediction_frames * 0.1)
                max_y = self.config.max_y_jump * (1 + prediction_frames * 0.1)
            elif is_locked:
                # Locked player: stricter
                max_dist = self.config.max_reid_dist * 0.7
                max_x = self.config.max_x_jump * 0.7
                max_y = self.config.max_y_jump * 0.7
            else:
                # Normal tracking
                max_dist = self.config.max_reid_dist
                max_x = self.config.max_x_jump
                max_y = self.config.max_y_jump

            # Evaluate each detection
            for di, det in enumerate(detections):
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = det["center"]
                det_color_cat = det["color_category"]

                # 1) Spatial jump check
                dx, dy = compute_spatial_jump(last_center, (cx, cy))

                if prediction_frames == 0:
                    # Active tracking: strict spatial constraints
                    if dx > max_x or dy > max_y:
                        continue
                else:
                    # Prediction mode: more lenient
                    if dx > max_x * 2 or dy > max_y * 2:
                        continue

                # 2) Distance from prediction
                d = (cx - pred_x) ** 2 + (cy - pred_y) ** 2
                if d > max_dist * (1 + prediction_frames * 0.2):
                    continue

                # 3) Color category match
                color_cat_penalty = ColorCategory.get_similarity_penalty(
                    player.color_category, det_color_cat
                )

                # Locked players: strict color matching
                if is_locked and prediction_frames == 0 and color_cat_penalty > 10:
                    continue

                # 4) IoU check
                iou = bbox_iou(last_bbox, (x1, y1, x2, y2))
                if is_locked and prediction_frames == 0 and iou < self.config.min_iou_for_match:
                    continue

                # 5) Velocity consistency
                vel_diff = compute_velocity_consistency(player.kalman, cx, cy)
                max_vel = self.config.max_velocity_change * (1 + prediction_frames * 0.1)
                if vel_diff > max_vel and prediction_frames == 0:
                    continue

                # 6) Size consistency
                size_diff = compute_size_consistency(
                    last_bbox, (x1, y1, x2, y2),
                    player.avg_bbox_size,
                    self.config.max_size_change
                )

                # 7) Direction consistency
                dir_penalty = compute_direction_consistency(
                    player.position_history, cx, cy
                )

                # 8) Appearance distance
                shirt_dist = AppearanceModel.compute_appearance_distance(
                    player.shirt_hist, det["shirt_hist"]
                )
                pants_dist = AppearanceModel.compute_appearance_distance(
                    player.pants_hist, det["pants_hist"]
                )

                # 9) Compute total cost
                cost = 0.0

                # IoU bonus (negative cost)
                cost -= iou * 15.0

                # Distance penalty
                cost += np.sqrt(d) * self.config.distance_penalty

                # Spatial penalty
                spatial_penalty = (
                    dx / self.config.max_x_jump +
                    dy / self.config.max_y_jump
                ) * self.config.spatial_penalty
                cost += spatial_penalty

                # Appearance costs
                cost += color_cat_penalty
                cost += shirt_dist * self.config.shirt_weight
                cost += pants_dist * self.config.pants_weight

                # Motion costs
                cost += vel_diff * 0.05 * self.config.velocity_weight
                cost += dir_penalty * self.config.direction_weight
                cost += size_diff * self.config.size_weight

                # Stability bonus
                cost -= player.stability * 0.2

                # Locked player bonus
                if is_locked:
                    cost -= 5.0

                # Lost player recovery bonus
                if prediction_frames > 0:
                    cost -= 3.0

                # Consecutive matches bonus
                if player.consecutive_matches > 3:
                    cost -= min(player.consecutive_matches * 0.4, 3.0)

                cost_matrix[pi, di] = cost

        return cost_matrix, player_ids, detections

    def match_players_to_detections(self, detections: List[dict],
                                   frame_width: int,
                                   frame_height: int) -> Tuple[
        List[Tuple[int, int]], set, set
    ]:
        """
        Match players to detections using Hungarian algorithm.

        Args:
            detections: List of detection dictionaries
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            (matched_pairs, matched_player_ids, matched_detection_indices)
            where matched_pairs is a list of (player_id, detection_idx) tuples
        """
        matched_pairs = []
        matched_players = set()
        matched_dets = set()

        if len(detections) == 0:
            return matched_pairs, matched_players, matched_dets

        # Build cost matrix
        cost_matrix, player_ids, det_list = self.build_cost_matrix(
            detections, frame_width, frame_height
        )

        if cost_matrix is None:
            return matched_pairs, matched_players, matched_dets

        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Process matches
        for row_idx, col_idx in zip(row_indices, col_indices):
            cost = cost_matrix[row_idx, col_idx]

            # Check if match is valid (cost threshold)
            if cost < 1e5:
                player_id = player_ids[row_idx]
                matched_pairs.append((player_id, col_idx))
                matched_players.add(player_id)
                matched_dets.add(col_idx)

        return matched_pairs, matched_players, matched_dets

    def update_with_detections(self, frame: np.ndarray,
                               detections: List[dict],
                               frame_idx: int,
                               frame_width: int,
                               frame_height: int):
        """
        Update all players with new detections.

        Args:
            frame: Current frame
            detections: List of detections
            frame_idx: Current frame index
            frame_width: Frame width
            frame_height: Frame height
        """
        # Match players to detections
        matched_pairs, matched_players, matched_dets = self.match_players_to_detections(
            detections, frame_width, frame_height
        )

        # Update matched players using the pairs directly
        for player_id, det_idx in matched_pairs:
            det = detections[det_idx]
            player = self.players[player_id]

            player.update_with_detection(
                bbox=det["bbox"],
                center=det["center"],
                frame_idx=frame_idx,
                shirt_hist=det["shirt_hist"],
                pants_hist=det["pants_hist"],
                hist_alpha=self.config.hist_alpha
            )

        # Update unmatched players with prediction
        for player_id, player in self.players.items():
            if player_id not in matched_players:
                player.update_with_prediction(
                    frame_width, frame_height,
                    self.config.max_prediction_frames
                )

    def get_active_count(self) -> int:
        """Get number of actively tracked players."""
        return sum(1 for p in self.players.values() if p.is_active())

    def get_predicted_count(self) -> int:
        """Get number of players in prediction mode."""
        return sum(1 for p in self.players.values() if not p.is_active())

    def get_statistics(self) -> dict:
        """
        Get tracking statistics.

        Returns:
            Dictionary with tracking stats
        """
        return {
            'total_players': len(self.players),
            'active_players': self.get_active_count(),
            'predicted_players': self.get_predicted_count(),
            'avg_stability': np.mean([p.stability for p in self.players.values()]),
        }

    def __len__(self) -> int:
        return len(self.players)

    def __getitem__(self, player_id: int) -> Player:
        return self.players[player_id]

    def items(self):
        return self.players.items()

    def values(self):
        return self.players.values()
