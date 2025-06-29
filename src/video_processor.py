"""
Video processing module for climbing motion tracking.
Handles video I/O, frame processing, and video reconstruction.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator
from pathlib import Path
import os
from tqdm import tqdm


class VideoProcessor:
    """Handles video input/output and frame processing."""
    
    def __init__(self):
        """Initialize the video processor."""
        self.input_video: Optional[cv2.VideoCapture] = None
        self.output_video: Optional[cv2.VideoWriter] = None
        self.fps = 0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.fourcc = None
        
    def open_input_video(self, video_path: str) -> bool:
        """
        Open input video file.
        
        Args:
            video_path: Path to input video file
            
        Returns:
            True if video opened successfully, False otherwise
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return False
            
        self.input_video = cv2.VideoCapture(video_path)
        
        if not self.input_video.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return False
            
        # Get video properties
        self.fps = self.input_video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video loaded: {self.width}x{self.height} @ {self.fps:.2f} fps, {self.frame_count} frames")
        return True
    
    def prepare_output_video(self, output_path: str, codec: str = 'mp4v') -> bool:
        """
        Prepare output video writer.
        
        Args:
            output_path: Path for output video file
            codec: Video codec to use
            
        Returns:
            True if output video prepared successfully, False otherwise
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Set up video writer
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.output_video = cv2.VideoWriter(
            output_path, 
            self.fourcc, 
            self.fps, 
            (self.width, self.height)
        )
        
        if not self.output_video.isOpened():
            print(f"Error: Could not create output video: {output_path}")
            return False
            
        return True
    
    def get_frames(self) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Generator to yield frames from input video.
        
        Yields:
            Tuple of (frame, frame_number, timestamp)
        """
        if self.input_video is None:
            return
            
        frame_number = 0
        
        while True:
            ret, frame = self.input_video.read()
            if not ret:
                break
                
            timestamp = frame_number / self.fps
            yield frame, frame_number, timestamp
            frame_number += 1
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the output video.
        
        Args:
            frame: Frame to write (BGR format)
            
        Returns:
            True if frame written successfully, False otherwise
        """
        if self.output_video is None:
            return False
            
        return self.output_video.write(frame)
    
    def close(self):
        """Close video files and release resources."""
        if self.input_video:
            self.input_video.release()
        if self.output_video:
            self.output_video.release()
        cv2.destroyAllWindows()
    
    def get_video_info(self) -> dict:
        """
        Get video information.
        
        Returns:
            Dictionary with video properties
        """
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration': self.frame_count / self.fps if self.fps > 0 else 0
        }


class FrameProcessor:
    """Handles frame-by-frame processing and pose data management."""
    
    def __init__(self):
        """Initialize the frame processor."""
        self.pose_frames: List = []
        self.processed_frames: List = []
        
    def add_pose_frame(self, pose_frame):
        """Add a pose frame to the processing queue."""
        self.pose_frames.append(pose_frame)
    
    def get_pose_history(self, current_frame: int, history_length: int = 30) -> List:
        """
        Get pose history for motion trails.
        
        Args:
            current_frame: Current frame number
            history_length: Number of previous frames to include
            
        Returns:
            List of pose frames for motion trails
        """
        start_idx = max(0, current_frame - history_length)
        return self.pose_frames[start_idx:current_frame]
    
    def clear_history(self):
        """Clear processed data to free memory."""
        self.pose_frames.clear()
        self.processed_frames.clear()


def resize_frame(frame: np.ndarray, target_width: Optional[int] = None, target_height: Optional[int] = None) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        target_width: Target width (if None, calculated from height)
        target_height: Target height (if None, calculated from width)
        
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    if target_width is None and target_height is None:
        return frame
    
    if target_width is None and target_height is not None:
        # Calculate width based on height
        aspect_ratio = width / height
        target_width = int(target_height * aspect_ratio)
    elif target_height is None and target_width is not None:
        # Calculate height based on width
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
    else:
        # Both provided, use as is
        target_width = target_width or width
        target_height = target_height or height
    
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def create_progress_bar(total_frames: int, description: str = "Processing") -> tqdm:
    """
    Create a progress bar for video processing.
    
    Args:
        total_frames: Total number of frames to process
        description: Description for the progress bar
        
    Returns:
        tqdm progress bar object
    """
    return tqdm(total=total_frames, desc=description, unit="frames") 