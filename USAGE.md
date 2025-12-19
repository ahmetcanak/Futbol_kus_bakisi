# Multi-Player Tracking System - Usage Guide

## 📁 Project Structure

```
player_tracking/
├── __init__.py          # Package initialization
├── config.py            # Configuration classes
├── player.py            # Player state management
├── appearance.py        # Appearance model (color, histogram)
├── tracker.py           # Multi-player tracker with Hungarian matching
├── video_processor.py   # Video I/O and visualization
└── utils.py             # Utility functions

main.py                  # Entry point
```

## 🚀 Quick Start

### Basic Usage

```bash
python main.py
```

This will use default settings:
- Input: `shs2.mp4`
- Output: `shs2_fixed_players_v5.mp4`
- Model: `yolov8m.pt`
- Players: 14

### Custom Configuration

```bash
python main.py --video path/to/video.mp4 \
               --output path/to/output.mp4 \
               --model yolov8x.pt \
               --players 10 \
               --conf 0.3
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video` | str | shs2.mp4 | Input video path |
| `--output` | str | shs2_fixed_players_v5.mp4 | Output video path |
| `--model` | str | yolov8m.pt | YOLO model path |
| `--players` | int | 14 | Number of players to track |
| `--conf` | float | 0.25 | YOLO confidence threshold |
| `--max-init-frames` | int | 60 | Max frames to search for initialization |

## 🔧 Advanced Usage

### Using as a Library

```python
from player_tracking import TrackingConfig, VideoProcessor

# Create custom configuration
config = TrackingConfig(
    video_path="my_video.mp4",
    output_path="output.mp4",
    fixed_player_count=12,
    yolo_conf_threshold=0.3,
    max_reid_dist=40**2,  # More lenient matching
)

# Create processor and run
processor = VideoProcessor(config)
processor.process_video()
```

### Customizing Tracking Parameters

```python
config = TrackingConfig(
    # Video settings
    video_path="video.mp4",
    output_path="output.mp4",

    # Detection
    yolo_model="yolov8x.pt",  # More accurate model
    yolo_conf_threshold=0.35,  # Higher confidence

    # Re-identification
    max_reid_dist=30**2,  # Stricter spatial matching
    max_prediction_frames=90,  # Allow longer prediction

    # Appearance weights
    shirt_weight=20.0,  # Increase shirt importance
    pants_weight=10.0,

    # Anti-swap parameters
    velocity_weight=10.0,  # Stricter velocity check
    direction_weight=8.0,
    size_weight=12.0,
)
```

### Bird's Eye View Calibration

Edit perspective transform points in config:

```python
config = TrackingConfig(
    # Your custom field corners
    src_points=np.float32([
        [x1, y1],  # Top-left
        [x2, y2],  # Bottom-left
        [x3, y3],  # Top-right
        [x4, y4],  # Bottom-right
        [x5, y5],  # Bottom-bottom
        [x6, y6],  # Top-bottom-right
    ]),

    # Target rectangle
    dst_points=np.float32([
        [50, 50],
        [50, 550],
        [350, 50],
        [350, 550],
        [200, 550],
        [350, 300],
    ]),

    # Real field dimensions
    field_width_m=20.0,   # 20 meters
    field_height_m=40.0,  # 40 meters
)
```

## 📊 Understanding the Output

### Video Output

The output video has two panels:

**Left Panel (Main View):**
- Bounding boxes around players
- Player IDs with labels
- Green box = Active tracking
- Orange dashed box = Prediction mode
- Stability score (S) or prediction frames (P)

**Right Panel (Bird's Eye View):**
- Top-down field view
- Solid circles = Active players
- Hollow circles = Predicted players
- Distance traveled (for Player ID 0)

### Console Output

```
Frame 1234/5000 (24.7%) - 45.2s - 27.3 FPS - Matched: 12/14
```

- Current frame / Total frames
- Progress percentage
- Elapsed time
- Processing FPS
- Matched players / Total players

### Final Statistics

```
Player Statistics:
  ID  0 [yellow       ]: Matched 4523/4940 frames (91.6%) - Distance: 523.4m
  ID  1 [black        ]: Matched 4321/4940 frames (87.5%) - Distance: 478.2m
  ...
```

## 🎨 Color Categories

The system recognizes these player categories:

- `black` - Dark uniforms (low brightness)
- `yellow` - Solid yellow jerseys
- `yellow_stripe` - Yellow/black striped pattern
- `green_vest` - Green training vests
- `white` - White or light colored
- `other` - Unclassified

## 🔍 Troubleshooting

### Issue: Players Not Initialized

**Problem:** "Could not find enough players for initialization"

**Solutions:**
1. Increase `max_init_frames`:
   ```bash
   python main.py --max-init-frames 120
   ```

2. Lower detection confidence:
   ```python
   config.yolo_conf_init = 0.15
   ```

3. Check if video starts with all players visible

### Issue: Frequent ID Swaps

**Problem:** Player IDs switching between players

**Solutions:**
1. Increase appearance weights:
   ```python
   config.shirt_weight = 25.0
   config.pants_weight = 15.0
   ```

2. Stricter spatial constraints:
   ```python
   config.max_x_jump = 100
   config.max_y_jump = 60
   ```

3. Lower max distance:
   ```python
   config.max_reid_dist = 30**2
   ```

### Issue: Players Lost After Occlusion

**Problem:** Players not recovered after being hidden

**Solutions:**
1. Increase prediction frames:
   ```python
   config.max_prediction_frames = 120
   ```

2. More lenient re-ID distance:
   ```python
   config.max_reid_dist_predicted = 80**2
   ```

3. Check histogram smoothing:
   ```python
   config.hist_alpha = 0.98  # More weight to original appearance
   ```

### Issue: Slow Processing

**Problem:** Low FPS during processing

**Solutions:**
1. Use smaller YOLO model:
   ```bash
   python main.py --model yolov8n.pt
   ```

2. Reduce image size:
   ```python
   config.yolo_imgsz = 960
   ```

3. Lower trajectory length:
   ```python
   config.trajectory_max_len = 150
   ```

## 📈 Performance Tuning

### For Speed (Higher FPS)

```python
config = TrackingConfig(
    yolo_model="yolov8n.pt",      # Fastest model
    yolo_imgsz=960,                 # Smaller resolution
    trajectory_max_len=100,         # Shorter history
    position_history_len=5,
)
```

### For Accuracy (Better Tracking)

```python
config = TrackingConfig(
    yolo_model="yolov8x.pt",        # Most accurate model
    yolo_imgsz=1920,                # Higher resolution
    yolo_conf_threshold=0.4,        # Higher confidence
    shirt_weight=25.0,              # Stronger appearance
    velocity_weight=12.0,           # Stricter motion
)
```

### For Crowded Scenes

```python
config = TrackingConfig(
    max_reid_dist=25**2,            # Stricter distance
    min_iou_for_match=0.25,         # Higher IoU requirement
    spatial_penalty=15.0,           # Penalize jumps more
    lock_threshold=8,               # More frames to lock
)
```

## 🧪 Testing Individual Components

### Test Player Detection

```python
from player_tracking import TrackingConfig, VideoProcessor
import cv2

config = TrackingConfig(video_path="test.mp4")
processor = VideoProcessor(config)
processor.cap = cv2.VideoCapture("test.mp4")

ret, frame = processor.cap.read()
detections = processor.detect_players(frame)
print(f"Found {len(detections)} players")
```

### Test Appearance Model

```python
from player_tracking import AppearanceModel
import cv2

model = AppearanceModel()
frame = cv2.imread("frame.jpg")
bbox = (100, 100, 200, 300)

color_cat, values = model.get_color_category(frame, bbox)
print(f"Color: {color_cat}, HSV: {values}")
```

### Test Bird's Eye View Transform

```python
from player_tracking import TrackingConfig
from player_tracking.utils import transform_point_to_bird_view

config = TrackingConfig()
matrix = config.get_homography_matrix()

# Test point transformation
camera_point = (640, 720)
bird_x, bird_y = transform_point_to_bird_view(camera_point, matrix)
print(f"Camera {camera_point} -> Bird ({bird_x}, {bird_y})")
```

## 📝 Configuration Reference

See `config.py` for all available parameters:

- **Detection**: `yolo_model`, `yolo_conf_threshold`, `yolo_imgsz`
- **Re-ID**: `max_reid_dist`, `max_prediction_frames`
- **Appearance**: `hist_size`, `shirt_weight`, `pants_weight`, `hist_alpha`
- **Anti-swap**: `velocity_weight`, `direction_weight`, `size_weight`
- **Spatial**: `max_x_jump`, `max_y_jump`, `spatial_penalty`
- **Stability**: `lock_threshold`, `position_history_len`
- **Bird View**: `bird_width`, `bird_height`, `field_width_m`, `field_height_m`

## 💡 Tips

1. **Always check the first frame**: Make sure all players are visible
2. **Tune for your specific video**: Different lighting/field requires different params
3. **Monitor match ratios**: < 80% means tracking issues
4. **Use GPU**: CUDA significantly speeds up processing
5. **Backup original appearance**: The system automatically saves initial histograms
