"""
Main entry point for multi-player tracking system.

Usage:
    python main.py

Or with custom config:
    python main.py --video path/to/video.mp4 --output path/to/output.mp4
"""

import argparse
from player_tracking import TrackingConfig, VideoProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-player tracking with fixed IDs"
    )

    parser.add_argument(
        "--video", type=str, default="shs2.mp4",
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", type=str, default="shs2_fixed_players_v5.mp4",
        help="Path to output video file"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8m.pt",
        help="YOLO model path"
    )
    parser.add_argument(
        "--players", type=int, default=14,
        help="Number of players to track"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--max-init-frames", type=int, default=60,
        help="Maximum frames to search for initialization"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Create configuration
    config = TrackingConfig(
        video_path=args.video,
        output_path=args.output,
        yolo_model=args.model,
        fixed_player_count=args.players,
        yolo_conf_threshold=args.conf,
        max_init_frames=args.max_init_frames,
    )

    print("="*70)
    print("MULTI-PLAYER TRACKING - FIXED IDs (Anti-Swap V5)")
    print("="*70)
    print(f"Video: {config.video_path}")
    print(f"Output: {config.output_path}")
    print(f"Model: {config.yolo_model}")
    print(f"Target players: {config.fixed_player_count}")
    print("="*70 + "\n")

    # Create processor and run
    processor = VideoProcessor(config)

    try:
        processor.process_video()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

    print("\n✅ Processing completed successfully!")


if __name__ == "__main__":
    main()
