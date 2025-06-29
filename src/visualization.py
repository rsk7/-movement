"""
Visualization module for climbing motion tracking.
Handles drawing pose overlays, motion trails, and joint angles.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import math
from dataclasses import dataclass

# Import pose detection classes
try:
    from .pose_detection import PoseFrame, PoseKeypoint
except ImportError:
    from pose_detection import PoseFrame, PoseKeypoint


class PoseVisualizer:
    """Handles visualization of pose data on video frames."""
    
    def __init__(self, 
                 skeleton_color: Tuple[int, int, int] = (0, 255, 0),
                 joint_color: Tuple[int, int, int] = (255, 0, 0),
                 trail_color: Tuple[int, int, int] = (0, 255, 255),
                 text_color: Tuple[int, int, int] = (255, 255, 255),
                 line_thickness: int = 2,
                 joint_radius: int = 4):
        """
        Initialize the pose visualizer.
        
        Args:
            skeleton_color: Color for skeleton lines (BGR)
            joint_color: Color for joint markers (BGR)
            trail_color: Color for motion trails (BGR)
            text_color: Color for text overlays (BGR)
            line_thickness: Thickness of skeleton lines
            joint_radius: Radius of joint markers
        """
        self.skeleton_color = skeleton_color
        self.joint_color = joint_color
        self.trail_color = trail_color
        self.text_color = text_color
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius
        
        # Define skeleton connections for climbing analysis
        self.skeleton_connections = [
            # Arms
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            
            # Shoulders
            ('left_shoulder', 'right_shoulder'),
            
            # Torso
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            
            # Legs
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
        ]
        
        # Define key joints for angle display
        self.angle_joints = {
            'left_shoulder': ('left_elbow', 'left_shoulder', 'left_wrist'),
            'right_shoulder': ('right_elbow', 'right_shoulder', 'right_wrist'),
            'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
            'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
            'left_hip': ('left_knee', 'left_hip', 'left_ankle'),
            'right_hip': ('right_knee', 'right_hip', 'right_ankle'),
        }
    
    def draw_pose_skeleton(self, frame: np.ndarray, pose_frame: PoseFrame) -> np.ndarray:
        """
        Draw skeleton overlay on frame.
        
        Args:
            frame: Input frame
            pose_frame: Pose data for the frame
            
        Returns:
            Frame with skeleton overlay
        """
        if not pose_frame or not pose_frame.keypoints:
            return frame
            
        height, width = frame.shape[:2]
        
        # Draw skeleton connections
        for connection in self.skeleton_connections:
            joint1_name, joint2_name = connection
            
            if joint1_name in pose_frame.keypoints and joint2_name in pose_frame.keypoints:
                joint1 = pose_frame.keypoints[joint1_name]
                joint2 = pose_frame.keypoints[joint2_name]
                
                # Convert normalized coordinates to pixel coordinates
                x1 = int(joint1.x * width)
                y1 = int(joint1.y * height)
                x2 = int(joint2.x * width)
                y2 = int(joint2.y * height)
                
                # Only draw if both points are visible
                if joint1.confidence > 0.3 and joint2.confidence > 0.3:
                    cv2.line(frame, (x1, y1), (x2, y2), self.skeleton_color, self.line_thickness)
        
        # Draw joint markers
        for joint_name, keypoint in pose_frame.keypoints.items():
            if keypoint.confidence > 0.3:
                x = int(keypoint.x * width)
                y = int(keypoint.y * height)
                cv2.circle(frame, (x, y), self.joint_radius, self.joint_color, -1)
        
        return frame
    
    def draw_motion_trails(self, frame: np.ndarray, pose_history: List[PoseFrame], 
                          trail_length: int = 30, fade_factor: float = 0.8) -> np.ndarray:
        """
        Draw motion trails showing recent movement history.
        
        Args:
            frame: Input frame
            pose_history: List of recent pose frames
            trail_length: Number of frames to show in trail
            fade_factor: Factor for trail fading (0-1)
            
        Returns:
            Frame with motion trails
        """
        if not pose_history or len(pose_history) < 2:
            return frame
            
        height, width = frame.shape[:2]
        
        # Limit trail length
        recent_frames = pose_history[-trail_length:]
        
        # Draw trails for key joints
        key_joints = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']
        
        for joint_name in key_joints:
            points = []
            
            # Collect points for this joint
            for pose_frame in recent_frames:
                if joint_name in pose_frame.keypoints:
                    keypoint = pose_frame.keypoints[joint_name]
                    if keypoint.confidence > 0.3:
                        x = int(keypoint.x * width)
                        y = int(keypoint.y * height)
                        points.append((x, y))
            
            # Draw trail with fading
            if len(points) > 1:
                for i in range(len(points) - 1):
                    alpha = (i / len(points)) * fade_factor
                    color = tuple(int(c * alpha) for c in self.trail_color)
                    cv2.line(frame, points[i], points[i + 1], color, max(1, self.line_thickness // 2))
        
        return frame
    
    def draw_joint_angles(self, frame: np.ndarray, pose_frame: PoseFrame, 
                         angles: Dict[str, float]) -> np.ndarray:
        """
        Draw joint angles on frame.
        
        Args:
            frame: Input frame
            pose_frame: Pose data for the frame
            angles: Dictionary of joint angles
            
        Returns:
            Frame with angle annotations
        """
        if not pose_frame or not pose_frame.keypoints:
            return frame
            
        height, width = frame.shape[:2]
        
        # Map angle keys to joint names for positioning
        angle_to_joint = {
            'left_shoulder_angle': 'left_shoulder',
            'right_shoulder_angle': 'right_shoulder',
            'left_elbow_angle': 'left_elbow',
            'right_elbow_angle': 'right_elbow',
            'left_hip_angle': 'left_hip',
            'right_hip_angle': 'right_hip',
            'left_knee_angle': 'left_knee',
            'right_knee_angle': 'right_knee',
        }
        
        # Draw angles for key joints
        for angle_name, angle_value in angles.items():
            if angle_name in angle_to_joint:
                joint_name = angle_to_joint[angle_name]
                if joint_name in pose_frame.keypoints:
                    keypoint = pose_frame.keypoints[joint_name]
                    if keypoint.confidence > 0.3:
                        x = int(keypoint.x * width)
                        y = int(keypoint.y * height)
                        
                        # Draw angle text
                        text = f"{angle_value:.1f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        
                        # Get text size for background
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        # Draw background rectangle
                        cv2.rectangle(frame, 
                                    (x - text_width//2 - 2, y - text_height - 2),
                                    (x + text_width//2 + 2, y + baseline + 2),
                                    (0, 0, 0), -1)
                        
                        # Draw text
                        cv2.putText(frame, text, 
                                  (x - text_width//2, y), 
                                  font, font_scale, self.text_color, thickness)
        
        return frame
    
    def draw_velocity_vectors(self, frame: np.ndarray, pose_history: List[PoseFrame], 
                            scale_factor: float = 50.0) -> np.ndarray:
        """
        Draw velocity vectors showing movement direction and speed.
        
        Args:
            frame: Input frame
            pose_history: List of recent pose frames
            scale_factor: Scale factor for velocity visualization
            
        Returns:
            Frame with velocity vectors
        """
        if len(pose_history) < 2:
            return frame
            
        height, width = frame.shape[:2]
        
        # Calculate velocity for key joints
        key_joints = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']
        
        for joint_name in key_joints:
            if len(pose_history) >= 2:
                current_frame = pose_history[-1]
                previous_frame = pose_history[-2]
                
                if (joint_name in current_frame.keypoints and 
                    joint_name in previous_frame.keypoints):
                    
                    current_kp = current_frame.keypoints[joint_name]
                    previous_kp = previous_frame.keypoints[joint_name]
                    
                    if current_kp.confidence > 0.3 and previous_kp.confidence > 0.3:
                        # Calculate velocity
                        dx = (current_kp.x - previous_kp.x) * width
                        dy = (current_kp.y - previous_kp.y) * height
                        
                        # Scale velocity for visualization
                        dx *= scale_factor
                        dy *= scale_factor
                        
                        # Current position
                        x = int(current_kp.x * width)
                        y = int(current_kp.y * height)
                        
                        # End position of velocity vector
                        end_x = int(x + dx)
                        end_y = int(y + dy)
                        
                        # Draw velocity vector
                        cv2.arrowedLine(frame, (x, y), (end_x, end_y), 
                                      (0, 255, 255), 2, tipLength=0.3)
        
        return frame
    
    def draw_performance_metrics(self, frame: np.ndarray, metrics: Dict[str, float]) -> np.ndarray:
        """
        Draw performance metrics on frame.
        
        Args:
            frame: Input frame
            metrics: Dictionary of performance metrics
            
        Returns:
            Frame with performance metrics
        """
        height, width = frame.shape[:2]
        
        # Draw metrics in top-left corner
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        for metric_name, value in metrics.items():
            text = f"{metric_name}: {value:.2f}"
            
            # Draw background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(frame, 
                        (10, y_offset - text_height - 5),
                        (10 + text_width + 10, y_offset + 5),
                        (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, text, (15, y_offset), font, font_scale, self.text_color, thickness)
            y_offset += 30
        
        return frame
    
    def create_overlay_frame(self, frame: np.ndarray, pose_frame: PoseFrame,
                           pose_history: Optional[List[PoseFrame]] = None,
                           angles: Optional[Dict[str, float]] = None,
                           show_trails: bool = True,
                           show_angles: bool = True,
                           show_velocity: bool = False,
                           trail_length: int = 30) -> np.ndarray:
        """
        Create a complete overlay frame with all visualizations.
        
        Args:
            frame: Input frame
            pose_frame: Current pose data
            pose_history: History of pose frames for trails
            angles: Joint angles to display
            show_trails: Whether to show motion trails
            show_angles: Whether to show joint angles
            show_velocity: Whether to show velocity vectors
            trail_length: Length of motion trails
            
        Returns:
            Frame with all overlays applied
        """
        # Make a copy to avoid modifying original
        overlay_frame = frame.copy()
        
        # Draw motion trails first (behind skeleton)
        if show_trails and pose_history:
            overlay_frame = self.draw_motion_trails(overlay_frame, pose_history, trail_length)
        
        # Draw skeleton
        overlay_frame = self.draw_pose_skeleton(overlay_frame, pose_frame)
        
        # Draw joint angles
        if show_angles and angles:
            overlay_frame = self.draw_joint_angles(overlay_frame, pose_frame, angles)
        
        # Draw velocity vectors
        if show_velocity and pose_history:
            overlay_frame = self.draw_velocity_vectors(overlay_frame, pose_history)
        
        return overlay_frame 