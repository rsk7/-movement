#!/usr/bin/env python3
"""
Main application for climbing motion tracking.
Orchestrates the entire pipeline from video input to tracked output.
"""

import argparse
import sys
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pose_detection import PoseDetector
from video_processor import VideoProcessor, FrameProcessor, create_progress_bar
from visualization import PoseVisualizer
from hold_detection import HoldDetector
from rhythm_detection import RhythmDetector


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
        self.hold_detector = HoldDetector()
        self.rhythm_detector = RhythmDetector()
        
        # Processing options
        self.show_trails = True
        self.show_angles = True
        self.show_velocity = False
        self.show_holds = False
        self.show_hold_contacts = False
        self.show_com = True
        self.show_rhythm = False
        self.trail_length = 30
        
    def process_video(self, input_path: str, output_path: str,
                     quality_factor: float = 1.0,
                     target_resolution: Optional[Tuple[int, int]] = None,
                     target_fps: Optional[float] = None,
                     overlay_only: bool = False) -> bool:
        """
        Process a climbing video and generate tracked output.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output video file
            quality_factor: Quality factor for processing (0.1-1.0)
            target_resolution: Target resolution for processing
            target_fps: Target frame rate for processing
            overlay_only: If True, output only pose overlays without background video
            
        Returns:
            True if processing successful, False otherwise
        """
        print(f"Processing video: {input_path}")
        print(f"Output will be saved to: {output_path}")
        if overlay_only:
            print("Mode: Overlay-only (no background video)")
        else:
            print("Mode: Full video with overlays")
        
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
            processed_frames = []  # Store processed frames for output generation
            
            # Use target resolution for frame processing
            processing_resolution = (self.video_processor.width, self.video_processor.height) if target_resolution else None
            
            pose_detected_count = 0
            total_frames = 0
            
            for frame, frame_number, timestamp in self.video_processor.get_frames(processing_resolution):
                # Store processed frame for output generation
                processed_frames.append(frame.copy())
                total_frames += 1
                
                # Detect pose in the frame
                pose_frame = self.pose_detector.detect_pose(frame, frame_number, timestamp)
                
                if pose_frame:
                    pose_frames.append(pose_frame)
                    self.frame_processor.add_pose_frame(pose_frame)
                    pose_detected_count += 1
                
                # Update progress
                progress_bar.update(1)
            
            progress_bar.close()
            
            print(f"Total frames processed: {total_frames}")
            print(f"Frames with pose detection: {pose_detected_count}")
            print(f"Pose detection rate: {pose_detected_count/total_frames*100:.1f}%")
            
            # Apply smoothing to pose data
            print("Applying pose smoothing...")
            smoothed_frames = self.pose_detector.smooth_pose_data(pose_frames)
            print(f"Smoothed pose frames: {len(smoothed_frames)}")
            
            # Process frames with visualizations
            print("Generating output video...")
            output_progress = create_progress_bar(len(processed_frames), "Generating output")
            
            # Create a mapping of frame numbers to pose frames
            pose_frame_map = {}
            for pose_frame in smoothed_frames:
                pose_frame_map[pose_frame.frame_number] = pose_frame
            
            print(f"Pose frame mapping: {len(pose_frame_map)} frames mapped")
            
            for i in range(len(processed_frames)):
                frame = processed_frames[i]
                
                # Get pose frame if available, otherwise use None
                pose_frame = pose_frame_map.get(i)
                
                if pose_frame:
                    # Calculate joint angles
                    angles = self.pose_detector.calculate_joint_angles(pose_frame)
                    
                    # Add frame to rhythm detector if COM is available
                    if self.show_rhythm and pose_frame.com is not None:
                        timestamp = i / video_info['fps']
                        self.rhythm_detector.add_frame(timestamp, pose_frame)
                        
                        # Get rhythm data for visualization
                        rhythm_summary = self.rhythm_detector.get_current_rhythm_summary()
                        rhythm_events = self.rhythm_detector.rhythm_events[-10:]  # Last 10 events
                        
                        pose_frame.rhythm_summary = rhythm_summary
                        pose_frame.rhythm_events = rhythm_events
                    
                    # Get pose history for trails using frame number mapping
                    pose_history = []
                    if self.show_trails:
                        # Get pose history by looking up previous frames in the mapping
                        for j in range(max(0, i - self.trail_length), i):
                            if j in pose_frame_map:
                                pose_history.append(pose_frame_map[j])
                    
                    # Detect holds in the frame
                    holds = []
                    hold_contacts = {}
                    if self.show_holds:
                        holds = self.hold_detector.detect_holds(frame)
                        if self.show_hold_contacts:
                            hold_contacts = self.hold_detector.detect_hold_contacts(pose_frame, holds)
                    
                    # Get original frame dimensions for correct landmark scaling
                    orig_width = self.video_processor.original_properties['width']
                    orig_height = self.video_processor.original_properties['height']
                    
                    if overlay_only:
                        # Create overlay-only frame (no background video)
                        overlay_frame = self.visualizer.create_overlay_only_frame(
                            width=orig_width,
                            height=orig_height,
                            pose_frame=pose_frame,
                            pose_history=pose_history,
                            angles=angles,
                            show_trails=self.show_trails,
                            show_angles=self.show_angles,
                            show_velocity=self.show_velocity,
                            show_com=self.show_com,
                            show_rhythm=self.show_rhythm,
                            trail_length=self.trail_length
                        )
                    else:
                        # Create overlay frame with original dimensions for correct landmark positioning
                        overlay_frame = self.visualizer.create_overlay_frame(
                            frame=frame,
                            pose_frame=pose_frame,
                            pose_history=pose_history,
                            angles=angles,
                            show_trails=self.show_trails,
                            show_angles=self.show_angles,
                            show_velocity=self.show_velocity,
                            show_com=self.show_com,
                            show_rhythm=self.show_rhythm,
                            trail_length=self.trail_length,
                            output_width=orig_width,
                            output_height=orig_height,
                            holds=holds,
                            hold_contacts=hold_contacts,
                            show_holds=self.show_holds,
                            show_hold_contacts=self.show_hold_contacts
                        )
                else:
                    # No pose detected
                    orig_width = self.video_processor.original_properties['width']
                    orig_height = self.video_processor.original_properties['height']
                    
                    if overlay_only:
                        # Create blank frame for overlay-only mode
                        overlay_frame = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
                    else:
                        # No pose detected, just resize the frame to output resolution
                        overlay_frame = cv2.resize(frame, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
                
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
                                 show_holds: bool = False,
                                 show_hold_contacts: bool = False,
                                 show_com: bool = True,
                                 show_rhythm: bool = False,
                                 trail_length: int = 30):
        """
        Set visualization options.
        
        Args:
            show_trails: Whether to show motion trails
            show_angles: Whether to show joint angles
            show_velocity: Whether to show velocity vectors
            show_holds: Whether to show detected holds
            show_hold_contacts: Whether to show hold contacts
            show_com: Whether to show center of mass
            show_rhythm: Whether to show rhythm analysis
            trail_length: Length of motion trails
        """
        self.show_trails = show_trails
        self.show_angles = show_angles
        self.show_velocity = show_velocity
        self.show_holds = show_holds
        self.show_hold_contacts = show_hold_contacts
        self.show_com = show_com
        self.show_rhythm = show_rhythm
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
    parser.add_argument('--show-holds', action='store_true',
                       help='Show detected climbing holds')
    parser.add_argument('--show-hold-contacts', action='store_true',
                       help='Highlight holds being contacted by climber')
    parser.add_argument('--show-com', action='store_true',
                       help='Show center of mass')
    parser.add_argument('--show-rhythm', action='store_true',
                       help='Show rhythm analysis')
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
    
    # Overlay-only mode
    parser.add_argument('--overlay-only', action='store_true',
                       help='Output only pose overlays without background video')
    
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
        show_holds=args.show_holds,
        show_hold_contacts=args.show_hold_contacts,
        show_com=args.show_com,
        show_rhythm=args.show_rhythm,
        trail_length=args.trail_length
    )
    
    # Process video
    success = tracker.process_video(args.input, args.output,
                                   args.quality_factor,
                                   (args.target_width, args.target_height) if args.target_width and args.target_height else None,
                                   args.target_fps,
                                   args.overlay_only)
    
    if success:
        print(f"\n✅ Processing completed!")
        print(f"Output video saved to: {args.output}")
    else:
        print("\n❌ Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 