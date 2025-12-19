"""
Video processor for multi-player tracking with bird's eye view visualization.
"""

import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from typing import List, Tuple, Optional
from .config import TrackingConfig
from .tracker import MultiPlayerTracker
from .appearance import AppearanceModel
from .utils import clamp_bbox_to_frame, transform_point_to_bird_view


class VideoProcessor:
    """
    Handles video I/O, YOLO detection, and visualization pipeline.
    """

    def __init__(self, config: TrackingConfig):
        """
        Initialize video processor.

        Args:
            config: Tracking configuration
        """
        self.config = config
        self.tracker = MultiPlayerTracker(config)
        self.appearance_model = AppearanceModel(
            hist_size=config.hist_size,
            hist_range=config.hist_range
        )

        # Initialize YOLO
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo = YOLO(config.yolo_model)
        self.yolo.to(self.device)

        # Video capture
        self.cap = None
        self.out = None
        self.fps = None
        self.width = None
        self.height = None
        self.total_frames = None

        # Homography matrix for bird's eye view
        self.homography_matrix = config.get_homography_matrix()

        print(f"Device: {self.device}")

    def find_initialization_frame(self) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[int]
    ]:
        """
        Find a frame with target number of players for initialization.

        Returns:
            (init_frame, init_boxes, frame_idx) or (None, None, None)
        """
        print(f"Searching for frame with {self.config.fixed_player_count} players...")

        best_frame = None
        best_boxes = None
        best_count = 0
        best_frame_idx = None

        for frame_idx in range(self.config.max_init_frames):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Run YOLO with lower confidence for initialization
            results = self.yolo(
                frame,
                classes=[0],  # Person class
                conf=self.config.yolo_conf_init,
                iou=self.config.yolo_iou_threshold,
                imgsz=self.config.yolo_imgsz,
                verbose=False,
                device=self.device
            )[0]

            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                count = len(boxes)

                print(f"  Frame {frame_idx}: {count} person(s) detected")

                # Update best frame
                if count > best_count:
                    best_count = count
                    best_frame = frame.copy()
                    best_boxes = boxes.copy()
                    best_frame_idx = frame_idx

                # If exact match found, use it immediately
                if count == self.config.fixed_player_count:
                    print(f"✓ Found exactly {self.config.fixed_player_count} "
                          f"players at frame {frame_idx}")
                    return frame.copy(), boxes, frame_idx

        # Use best frame if close enough
        if best_frame is not None and best_count >= self.config.fixed_player_count - 2:
            # If more detections than needed, select largest ones
            from .utils import bbox_area
            areas = [bbox_area(b) for b in best_boxes]
            sorted_indices = np.argsort(areas)[::-1]  # Descending
            selected_count = min(best_count, self.config.fixed_player_count)
            selected_indices = sorted_indices[:selected_count]
            init_boxes = best_boxes[selected_indices]

            print(f"⚠ Using best frame {best_frame_idx} with {best_count} players "
                  f"(taking {len(init_boxes)})")
            return best_frame, init_boxes, best_frame_idx

        return None, None, None

    def initialize_tracking(self) -> int:
        """
        Initialize video and tracker.

        Returns:
            Frame index where tracking starts

        Raises:
            RuntimeError if initialization fails
        """
        self.cap = cv2.VideoCapture(self.config.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.config.video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {self.width}x{self.height} @ {self.fps:.1f} fps, "
              f"{self.total_frames} frames")

        # Find initialization frame
        init_frame, init_boxes, init_frame_idx = self.find_initialization_frame()

        if init_frame is None:
            self.cap.release()
            raise RuntimeError(f"Could not find enough players for initialization")

        # Initialize tracker
        player_count = self.tracker.initialize_players(
            init_frame, init_boxes, init_frame_idx
        )

        print(f"\n✓ Initialized {player_count} players at frame {init_frame_idx}")

        # Restart video from initialization frame
        self.cap.release()
        self.cap = cv2.VideoCapture(self.config.video_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame_idx)

        # Initialize video writer
        output_width = self.width + self.config.bird_width
        output_height = max(self.height, self.config.bird_height)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            self.config.output_path, fourcc, self.fps,
            (output_width, output_height)
        )

        return init_frame_idx

    def detect_players(self, frame: np.ndarray) -> List[dict]:
        """
        Detect players in frame using YOLO.

        Args:
            frame: Input frame

        Returns:
            List of detection dictionaries
        """
        results = self.yolo(
            frame,
            classes=[0],  # Person class
            conf=self.config.yolo_conf_threshold,
            iou=self.config.yolo_iou_threshold,
            imgsz=self.config.yolo_imgsz,
            verbose=False,
            device=self.device
        )[0]

        detections = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1, x2, y2 = clamp_bbox_to_frame(
                    (x1, y1, x2, y2), self.width, self.height
                )
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Extract appearance features
                shirt_hist, pants_hist = self.appearance_model.get_shirt_pants_hist(
                    frame, (x1, y1, x2, y2)
                )
                color_cat, color_vals = self.appearance_model.get_color_category(
                    frame, (x1, y1, x2, y2)
                )

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "shirt_hist": shirt_hist,
                    "pants_hist": pants_hist,
                    "color_category": color_cat,
                    "color_values": color_vals,
                })

        return detections

    def create_bird_view(self, frame: np.ndarray) -> np.ndarray:
        """
        Create bird's eye view of the field.

        Args:
            frame: Input frame

        Returns:
            Bird's eye view frame
        """
        bird_view = cv2.warpPerspective(
            frame, self.homography_matrix,
            (self.config.bird_width, self.config.bird_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(30, 30, 30)
        )

        # Add field dimensions text
        cv2.putText(
            bird_view,
            f"Field: {self.config.field_width_m:.0f}m x {self.config.field_height_m:.0f}m",
            (10, self.config.bird_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        return bird_view

    def draw_player_on_frame(self, frame: np.ndarray, player):
        """
        Draw player bbox and label on frame.

        Args:
            frame: Frame to draw on
            player: Player instance
        """
        x1, y1, x2, y2 = player.last_bbox
        color = player.get_display_color()
        label = player.get_label()

        if player.is_active():
            # Active player: solid box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            bg_color = color
            text_color = (255, 255, 255)
        else:
            # Prediction mode: dashed effect
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), (0, 165, 255), 1)
            bg_color = (0, 165, 255)
            text_color = (0, 0, 0)

        # Draw label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - 25),
                     (x1 + label_size[0] + 8, y1), bg_color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 7),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

    def draw_player_on_bird_view(self, bird_view: np.ndarray, player,
                                 bird_x: int, bird_y: int):
        """
        Draw player on bird's eye view.

        Args:
            bird_view: Bird view frame
            player: Player instance
            bird_x: X coordinate in bird view
            bird_y: Y coordinate in bird view
        """
        if not (0 <= bird_x < self.config.bird_width and
                0 <= bird_y < self.config.bird_height):
            return

        color = player.get_display_color()

        if player.is_active():
            # Solid circle
            cv2.circle(bird_view, (bird_x, bird_y), 10, color, -1)
            cv2.circle(bird_view, (bird_x, bird_y), 10, (255, 255, 255), 2)
            text_color = (0, 0, 0)
        else:
            # Hollow circle for predicted
            cv2.circle(bird_view, (bird_x, bird_y), 8, color, 2)
            text_color = color

        # Draw ID
        cv2.putText(bird_view, str(player.id), (bird_x - 8, bird_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    def draw_info_overlay(self, frame: np.ndarray, bird_view: np.ndarray,
                         frame_idx: int, detections_count: int):
        """
        Draw information overlay on frames.

        Args:
            frame: Main frame
            bird_view: Bird view frame
            frame_idx: Current frame index
            detections_count: Number of detections
        """
        # Frame info
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Tracking stats
        active_count = self.tracker.get_active_count()
        predicted_count = self.tracker.get_predicted_count()

        cv2.putText(frame, f"Matched: {active_count} | Predicted: {predicted_count}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Detections: {detections_count}",
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Distance info on bird view (for player 0 if exists)
        if 0 in self.tracker.players:
            player = self.tracker.players[0]
            cv2.putText(bird_view, f"ID 0 Distance: {player.total_distance:.1f}m",
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    def process_video(self):
        """
        Main video processing loop.
        """
        # Initialize
        init_frame_idx = self.initialize_tracking()
        frame_idx = init_frame_idx - 1

        print(f"\nConfigured fixed players: {self.config.fixed_player_count}")
        print(f"Initialized players: {len(self.tracker)}")
        print("Processing started...\n")

        t0 = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_idx += 1

            # Progress update
            if frame_idx % 60 == 0:
                elapsed = time.time() - t0
                fps_actual = (frame_idx - init_frame_idx + 1) / elapsed
                matched = self.tracker.get_active_count()
                print(f"Frame {frame_idx}/{self.total_frames} "
                      f"({100*frame_idx/self.total_frames:.1f}%) - "
                      f"{elapsed:.1f}s - {fps_actual:.1f} FPS - "
                      f"Matched: {matched}/{len(self.tracker)}")

            # Detect players
            detections = self.detect_players(frame)

            # Create bird view
            bird_view = self.create_bird_view(frame)

            # Predict all players
            self.tracker.predict_all()

            # Update tracker
            self.tracker.update_with_detections(
                frame, detections, frame_idx, self.width, self.height
            )

            # Visualize all players
            for player in self.tracker.values():
                # Draw on main frame
                self.draw_player_on_frame(frame, player)

                # Transform to bird view
                cx, cy = player.last_center
                foot_x = cx
                foot_y = player.last_bbox[3]  # Bottom of bbox

                bird_x, bird_y = transform_point_to_bird_view(
                    (foot_x, foot_y), self.homography_matrix
                )

                # Update bird position and distance
                player.update_bird_view_position(
                    (bird_x, bird_y),
                    self.config.pixels_per_meter_x,
                    self.config.pixels_per_meter_y
                )

                # Draw on bird view
                self.draw_player_on_bird_view(bird_view, player, bird_x, bird_y)

            # Draw info overlay
            self.draw_info_overlay(frame, bird_view, frame_idx, len(detections))

            # Combine frames
            output_width = self.width + self.config.bird_width
            output_height = max(self.height, self.config.bird_height)
            combined = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            combined[:self.height, :self.width] = frame
            combined[:self.config.bird_height,
                    self.width:self.width + self.config.bird_width] = bird_view

            # Write output
            self.out.write(combined)

        # Cleanup
        elapsed = time.time() - t0
        avg_fps = (frame_idx - init_frame_idx + 1) / elapsed if elapsed > 0 else 0.0

        self.cap.release()
        self.out.release()

        # Print summary
        self.print_summary(elapsed, avg_fps, init_frame_idx)

    def print_summary(self, elapsed: float, avg_fps: float, init_frame_idx: int):
        """
        Print tracking summary statistics.

        Args:
            elapsed: Total processing time
            avg_fps: Average FPS
            init_frame_idx: Initialization frame index
        """
        print("\n" + "="*70)
        print(f"✓ FIXED {len(self.tracker)} PLAYERS TRACKING COMPLETED")
        print(f"✓ Time: {elapsed:.1f}s ({avg_fps:.1f} FPS)")
        print(f"✓ Output: {self.config.output_path}")
        print("\nPlayer Statistics:")
        print("-" * 70)

        for player in self.tracker.values():
            match_ratio = 100 * player.found_frames / max(1, (
                self.total_frames - init_frame_idx
            ))
            print(f"  ID {player.id:2d} [{player.color_category_original:13s}]: "
                  f"Matched {player.found_frames:4d}/"
                  f"{self.total_frames - init_frame_idx} frames "
                  f"({match_ratio:5.1f}%) - Distance: {player.total_distance:.1f}m")

        print("="*70)
