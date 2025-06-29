#!/usr/bin/env python3
"""
Live Pose Detection Script
Uses webcam for real-time pose detection and joint angle calculation.
"""

import cv2
import numpy as np
import argparse
import sys
import os
from typing import Dict, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pose_detection import PoseDetector, PoseFrame
from visualization import PoseVisualizer


class LivePoseDetector:
    """Real-time pose detection using webcam."""
    
    def __init__(self, 
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 show_angles: bool = True,
                 show_skeleton: bool = True,
                 camera_id: int = 0):
        """
        Initialize live pose detector.
        
        Args:
            model_complexity: MediaPipe model complexity (0, 1, 2)
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            show_angles: Whether to show joint angles
            show_skeleton: Whether to show skeleton overlay
            camera_id: Webcam device ID
        """
        self.pose_detector = PoseDetector(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=True
        )
        
        self.visualizer = PoseVisualizer()
        self.show_angles = show_angles
        self.show_skeleton = show_skeleton
        self.camera_id = camera_id
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Get camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {self.width}x{self.height} @ {self.fps:.1f} fps")
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = cv2.getTickCount()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for pose detection and visualization.
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Processed frame with overlays
        """
        # Detect pose
        pose_frame = self.pose_detector.detect_pose(frame, self.frame_count, 0.0)
        
        # Create output frame
        output_frame = frame.copy()
        
        if pose_frame and pose_frame.keypoints:
            # Calculate joint angles
            angles = self.pose_detector.calculate_joint_angles(pose_frame)
            
            # Draw skeleton
            if self.show_skeleton:
                output_frame = self.visualizer.draw_pose_skeleton(output_frame, pose_frame)
            
            # Draw joint angles
            if self.show_angles and angles:
                output_frame = self.visualizer.draw_joint_angles(output_frame, pose_frame, angles)
        
        # Draw FPS counter
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update FPS every 30 frames
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - self.fps_start_time) / cv2.getTickFrequency()
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        # Draw FPS on frame
        if hasattr(self, 'current_fps'):
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(output_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            "Press 'q' to quit",
            "Press 'a' to toggle angles",
            "Press 's' to toggle skeleton",
            "Press 'r' to reset pose"
        ]
        
        y_offset = 60
        for instruction in instructions:
            cv2.putText(output_frame, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        self.frame_count += 1
        return output_frame
    
    def run(self):
        """Main loop for live pose detection."""
        print("Starting live pose detection...")
        print("Press 'q' to quit, 'a' to toggle angles, 's' to toggle skeleton")
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Process frame
                output_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow('Live Pose Detection', output_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    self.show_angles = not self.show_angles
                    print(f"Angles display: {'ON' if self.show_angles else 'OFF'}")
                elif key == ord('s'):
                    self.show_skeleton = not self.show_skeleton
                    print(f"Skeleton display: {'ON' if self.show_skeleton else 'OFF'}")
                elif key == ord('r'):
                    # Reset pose detector
                    self.pose_detector.pose_history.clear()
                    print("Pose history reset")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        if self.cap:
            self.cap.release()
        self.pose_detector.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live Pose Detection - Real-time pose tracking with webcam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_pose_detection.py
  python live_pose_detection.py --camera 1 --model-complexity 2
  python live_pose_detection.py --no-angles --no-skeleton
        """
    )
    
    # Camera options
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='Camera device ID (default: 0)')
    
    # Detection options
    parser.add_argument('--model-complexity', type=int, choices=[0, 1, 2], default=1,
                       help='MediaPipe model complexity (default: 1)')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5,
                       help='Minimum confidence for pose detection (default: 0.5)')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5,
                       help='Minimum confidence for pose tracking (default: 0.5)')
    
    # Display options
    parser.add_argument('--no-angles', action='store_true',
                       help='Disable joint angle display')
    parser.add_argument('--no-skeleton', action='store_true',
                       help='Disable skeleton overlay')
    
    args = parser.parse_args()
    
    try:
        # Create and run live pose detector
        detector = LivePoseDetector(
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            show_angles=not args.no_angles,
            show_skeleton=not args.no_skeleton,
            camera_id=args.camera
        )
        
        detector.run()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 