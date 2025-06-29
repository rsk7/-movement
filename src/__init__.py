"""
Climbing Motion Tracker Package

A computer vision application for analyzing climbing videos with pose tracking.
"""

__version__ = "1.0.0"
__author__ = "Climbing Motion Tracker Team"

from .pose_detection import PoseDetector, PoseFrame, PoseKeypoint
from .video_processor import VideoProcessor, FrameProcessor
from .visualization import PoseVisualizer
from .main import ClimbingMotionTracker

__all__ = [
    'PoseDetector',
    'PoseFrame', 
    'PoseKeypoint',
    'VideoProcessor',
    'FrameProcessor',
    'PoseVisualizer',
    'ClimbingMotionTracker'
] 