#!/usr/bin/env python3
"""
Main application for climbing motion tracking.
Orchestrates the entire pipeline from video input to tracked output.
"""

import argparse
import sys
import os
import cv2
from pathlib import Path
from typing import Optional, Tuple

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pose_detection import PoseDetector
from video_processor import VideoProcessor, FrameProcessor, create_progress_bar
from visualization import PoseVisualizer


class ClimbingMotionTracker:
    """Main class for climbing motion tracking application."""
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 smooth_landmarks: bool = True):
        """
        Initialize the climbing motion tracker.
        
        Args:
            model_complexity: MediaPipe model complexity (0, 1, or 2)
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            smooth_landmarks: Whether to smooth landmarks across frames
        """
        self.pose_detector = PoseDetector(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=smooth_landmarks
        )
        
        self.video_processor = VideoProcessor()
        self.frame_processor = FrameProcessor()
        self.visualizer = PoseVisualizer()
        
        # Processing options
        self.show_trails = True
        self.show_angles = True
        self.show_velocity = False
        self.trail_length = 30
        
    def process_video(self, input_path: str, output_path: str,
                     quality_factor: float = 1.0,
                     target_resolution: Optional[Tuple[int, int]] = None,
                     target_fps: Optional[float] = None) -> bool:
        """
        Process a climbing video and generate tracked output.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            quality_factor: Quality factor for processing (0.1-1.0)
            target_resolution: Target resolution for processing
            target_fps: Target frame rate for processing
            
        Returns:
            True if processing successful, False otherwise
        """
        print(f"Processing video: {input_path}")
        print(f"Output will be saved to: {output_path}")
        
        # Open input video with preprocessing options
        if not self.video_processor.open_input_video(
            input_path, 
            target_resolution=target_resolution,
            target_fps=target_fps,
            quality_factor=quality_factor
        ):
            return False
        
        # Prepare output video
        if not self.video_processor.prepare_output_video(output_path):
            return False
        
        # Get video info
        video_info = self.video_processor.get_video_info()
        print(f"Video duration: {video_info['duration']:.2f} seconds")
        
        # Create progress bar
        progress_bar = create_progress_bar(video_info['frame_count'], "Processing frames")
        
        try:
            # Process frames
            pose_frames = []
            
            # Use target resolution for frame processing
            processing_resolution = (self.video_processor.width, self.video_processor.height) if target_resolution else None
            
            for frame, frame_number, timestamp in self.video_processor.get_frames(processing_resolution):
                # Detect pose in frame
                pose_frame = self.pose_detector.detect_pose(frame, frame_number, timestamp)
                
                if pose_frame:
                    pose_frames.append(pose_frame)
                    self.frame_processor.add_pose_frame(pose_frame)
                
                # Update progress
                progress_bar.update(1)
            
            progress_bar.close()
            
            # Apply smoothing to pose data
            print("Applying pose smoothing...")
            smoothed_frames = self.pose_detector.smooth_pose_data(pose_frames)
            
            # Process frames with visualizations
            print("Generating output video...")
            output_progress = create_progress_bar(len(smoothed_frames), "Generating output")
            
            for i, pose_frame in enumerate(smoothed_frames):
                # Get original frame
                if self.video_processor.input_video is not None:
                    self.video_processor.input_video.set(cv2.CAP_PROP_POS_FRAMES, pose_frame.frame_number)
                    ret, frame = self.video_processor.input_video.read()
                    
                    if ret:
                        # Resize frame to match VideoWriter's expected dimensions
                        frame = cv2.resize(frame, (self.video_processor.width, self.video_processor.height), interpolation=cv2.INTER_AREA)
                        
                        # Calculate joint angles
                        angles = self.pose_detector.calculate_joint_angles(pose_frame)
                        
                        # Get pose history for trails
                        pose_history = self.frame_processor.get_pose_history(i, self.trail_length)
                        
                        # Create overlay frame
                        overlay_frame = self.visualizer.create_overlay_frame(
                            frame=frame,
                            pose_frame=pose_frame,
                            pose_history=pose_history,
                            angles=angles,
                            show_trails=self.show_trails,
                            show_angles=self.show_angles,
                            show_velocity=self.show_velocity,
                            trail_length=self.trail_length
                        )
                        
                        # Write frame to output
                        self.video_processor.write_frame(overlay_frame)
                
                output_progress.update(1)
            
            output_progress.close()
            
            print("Processing completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during processing: {e}")
            return False
        
        finally:
            # Clean up
            self.video_processor.close()
            self.pose_detector.release()
    
    def set_visualization_options(self, 
                                 show_trails: bool = True,
                                 show_angles: bool = True,
                                 show_velocity: bool = False,
                                 trail_length: int = 30):
        """
        Set visualization options.
        
        Args:
            show_trails: Whether to show motion trails
            show_angles: Whether to show joint angles
            show_velocity: Whether to show velocity vectors
            trail_length: Length of motion trails
        """
        self.show_trails = show_trails
        self.show_angles = show_angles
        self.show_velocity = show_velocity
        self.trail_length = trail_length


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Climbing Motion Tracker - Analyze climbing videos with pose tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input climbing_video.mp4 --output tracked_video.mp4
  python main.py --input video.mp4 --output output.mp4 --show-trails --show-angles
  python main.py --input video.mp4 --output output.mp4 --trail-length 50 --show-velocity
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Path to input climbing video file')
    parser.add_argument('--output', '-o', required=True,
                       help='Path for output video with tracking')
    
    # Visualization options
    parser.add_argument('--show-trails', action='store_true',
                       help='Show motion trails (movement history)')
    parser.add_argument('--show-angles', action='store_true',
                       help='Show joint angles on frame')
    parser.add_argument('--show-velocity', action='store_true',
                       help='Show velocity vectors')
    parser.add_argument('--trail-length', type=int, default=30,
                       help='Number of frames to show in motion trails (default: 30)')
    
    # Processing options
    parser.add_argument('--model-complexity', type=int, choices=[0, 1, 2], default=1,
                       help='MediaPipe model complexity (default: 1)')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5,
                       help='Minimum confidence for pose detection (default: 0.5)')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5,
                       help='Minimum confidence for pose tracking (default: 0.5)')
    parser.add_argument('--no-smoothing', action='store_true',
                       help='Disable pose smoothing')
    
    # Video preprocessing options
    parser.add_argument('--quality-factor', type=float, default=1.0,
                       help='Quality factor for processing (0.1-1.0, default: 1.0)')
    parser.add_argument('--target-width', type=int,
                       help='Target width for processing (maintains aspect ratio)')
    parser.add_argument('--target-height', type=int,
                       help='Target height for processing (maintains aspect ratio)')
    parser.add_argument('--target-fps', type=float,
                       help='Target frame rate for processing (default: original fps)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize tracker
    tracker = ClimbingMotionTracker(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        smooth_landmarks=not args.no_smoothing
    )
    
    # Set visualization options
    tracker.set_visualization_options(
        show_trails=args.show_trails,
        show_angles=args.show_angles,
        show_velocity=args.show_velocity,
        trail_length=args.trail_length
    )
    
    # Process video
    success = tracker.process_video(args.input, args.output,
                                   args.quality_factor,
                                   (args.target_width, args.target_height) if args.target_width and args.target_height else None,
                                   args.target_fps)
    
    if success:
        print(f"\n✅ Processing completed!")
        print(f"Output video saved to: {args.output}")
    else:
        print("\n❌ Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 