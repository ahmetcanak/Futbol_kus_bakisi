"""
Appearance model for player re-identification using color and histogram features.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .utils import compute_shirt_pants_bboxes
from .config import ColorCategory


class AppearanceModel:
    """
    Manages appearance features for player re-identification.
    Handles color categorization, histogram computation, and appearance matching.
    """

    def __init__(self, hist_size: Tuple[int, int] = (32, 32),
                 hist_range: list = None):
        """
        Initialize appearance model.

        Args:
            hist_size: Size of HSV histogram (H bins, S bins)
            hist_range: Range for histogram [H_min, H_max, S_min, S_max]
        """
        self.hist_size = hist_size
        self.hist_range = hist_range or [0, 180, 0, 256]

    def get_color_category(self, frame: np.ndarray,
                           bbox: Tuple[int, int, int, int]) -> Tuple[str, Optional[Tuple]]:
        """
        Determine player's color category based on shirt region.

        Args:
            frame: Input frame (BGR)
            bbox: Player bounding box (x1, y1, x2, y2)

        Returns:
            (color_category, (avg_h, avg_s, avg_v)) or (category, None)
        """
        x1, y1, x2, y2 = bbox
        h_img, w_img = frame.shape[:2]

        # Clamp bbox to frame
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img))

        if x2 <= x1 or y2 <= y1:
            return ColorCategory.OTHER, None

        h = y2 - y1
        w = x2 - x1

        # Extract shirt region (15-50% of height)
        shirt_y1 = y1 + int(h * 0.15)
        shirt_y2 = y1 + int(h * 0.50)
        margin = int(w * 0.15)
        shirt_x1 = x1 + margin
        shirt_x2 = x2 - margin

        if shirt_x2 <= shirt_x1 or shirt_y2 <= shirt_y1:
            return ColorCategory.OTHER, None

        crop = frame[shirt_y1:shirt_y2, shirt_x1:shirt_x2]
        if crop.size == 0:
            return ColorCategory.OTHER, None

        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # Create mask to filter out grass and invalid pixels
        mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255
        grass_mask = ((H >= 25) & (H <= 100) & (S >= 35))
        mask[grass_mask] = 0

        valid_pixels = cv2.countNonZero(mask)
        if valid_pixels < 30:
            return ColorCategory.OTHER, None

        # Calculate average HSV values
        masked_H = H[mask > 0]
        masked_S = S[mask > 0]
        masked_V = V[mask > 0]

        avg_h = float(np.mean(masked_H))
        avg_s = float(np.mean(masked_S))
        avg_v = float(np.mean(masked_V))

        h_std = float(np.std(masked_H))
        v_std = float(np.std(masked_V))

        # Count specific color pixels
        yellow_count = np.sum((masked_H >= 15) & (masked_H <= 35) & (masked_S >= 80))
        black_count = np.sum((masked_V < 60))
        total = len(masked_H)

        if total == 0:
            return ColorCategory.OTHER, (avg_h, avg_s, avg_v)

        yellow_ratio = yellow_count / total
        black_ratio = black_count / total

        # Classify based on color statistics
        # Yellow stripe pattern (high variance in both hue and value)
        if yellow_ratio > 0.2 and black_ratio > 0.15 and (h_std > 20 or v_std > 40):
            return ColorCategory.YELLOW_STRIPE, (avg_h, avg_s, avg_v)

        # Black (low value, low saturation)
        if avg_v < 70 and avg_s < 80:
            return ColorCategory.BLACK, (avg_h, avg_s, avg_v)

        # Yellow (yellow hue, high saturation, high value)
        if 15 <= avg_h <= 40 and avg_s >= 80 and avg_v >= 150:
            return ColorCategory.YELLOW, (avg_h, avg_s, avg_v)

        # Green vest (green hue, high saturation, high value)
        if 35 <= avg_h <= 80 and avg_s >= 100 and avg_v >= 150:
            return ColorCategory.GREEN_VEST, (avg_h, avg_s, avg_v)

        # White (low saturation, high value)
        if avg_s < 40 and avg_v > 180:
            return ColorCategory.WHITE, (avg_h, avg_s, avg_v)

        return ColorCategory.OTHER, (avg_h, avg_s, avg_v)

    def get_clean_hist(self, frame: np.ndarray,
                      bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Compute cleaned HSV histogram for a region, filtering out grass and noise.

        Args:
            frame: Input frame (BGR)
            bbox: Region bounding box (x1, y1, x2, y2)

        Returns:
            Normalized histogram or None if invalid
        """
        x1, y1, x2, y2 = bbox
        h_img, w_img = frame.shape[:2]

        # Clamp to frame
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 10:
            return None

        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # Create cleaning mask
        mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255

        # Filter grass
        grass_mask = ((H >= 25) & (H <= 100) & (S >= 35))
        mask[grass_mask] = 0

        # Filter very dark pixels
        dark_mask = (V < 20)
        mask[dark_mask] = 0

        # Filter very bright/white pixels
        bright_mask = (V > 250) & (S < 20)
        mask[bright_mask] = 0

        if cv2.countNonZero(mask) < 50:
            return None

        # Compute histogram
        hist = cv2.calcHist([hsv], [0, 1], mask, self.hist_size, self.hist_range)
        cv2.normalize(hist, hist)

        return hist

    def get_shirt_pants_hist(self, frame: np.ndarray,
                             bbox: Tuple[int, int, int, int]) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray]
    ]:
        """
        Compute separate histograms for shirt and pants regions.

        Args:
            frame: Input frame (BGR)
            bbox: Full person bbox (x1, y1, x2, y2)

        Returns:
            (shirt_hist, pants_hist) tuple
        """
        shirt_bbox, pants_bbox = compute_shirt_pants_bboxes(bbox)

        shirt_hist = None
        pants_hist = None

        if shirt_bbox is not None:
            shirt_hist = self.get_clean_hist(frame, shirt_bbox)

        if pants_bbox is not None:
            pants_hist = self.get_clean_hist(frame, pants_bbox)

        return shirt_hist, pants_hist

    @staticmethod
    def smooth_histogram(old_hist: Optional[np.ndarray],
                        new_hist: Optional[np.ndarray],
                        alpha: float = 0.95) -> Optional[np.ndarray]:
        """
        Smooth histogram update with exponential moving average.

        Args:
            old_hist: Previous histogram
            new_hist: New histogram
            alpha: Smoothing factor (higher = more weight to old)

        Returns:
            Smoothed histogram
        """
        if old_hist is None:
            return new_hist
        if new_hist is None:
            return old_hist

        smoothed = alpha * old_hist + (1 - alpha) * new_hist
        cv2.normalize(smoothed, smoothed)
        return smoothed

    @staticmethod
    def compute_appearance_distance(hist1: Optional[np.ndarray],
                                   hist2: Optional[np.ndarray]) -> float:
        """
        Compute appearance distance between two histograms.

        Args:
            hist1: First histogram
            hist2: Second histogram

        Returns:
            Distance value (0 = identical, 1 = completely different)
        """
        if hist1 is None or hist2 is None:
            return 0.5

        # Correlation (1 = identical, -1 = opposite)
        correl = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # Bhattacharyya distance (0 = identical, 1 = different)
        bhatta = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

        # Combine metrics
        distance = (1 - correl) * 0.5 + bhatta * 0.5
        return distance
