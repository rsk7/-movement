"""
Pose detection module using MediaPipe for climbing motion tracking.
Handles pose estimation, keypoint extraction, and data smoothing.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.signal import savgol_filter


@dataclass
class PoseKeypoint:
    """Represents a single pose keypoint with position and confidence."""
    x: float
    y: float
    z: float
    confidence: float


@dataclass
class PoseFrame:
    """Represents pose data for a single frame."""
    keypoints: Dict[str, PoseKeypoint]
    timestamp: float
    frame_number: int


class PoseDetector:
    """MediaPipe-based pose detector optimized for climbing videos."""
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the pose detector.
        
        Args:
            static_image_mode: Whether to process static images or video
            model_complexity: Model complexity (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks across frames
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose  # type: ignore
        self.mp_drawing = mp.solutions.drawing_utils  # type: ignore
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define key body parts for climbing analysis
        self.key_landmarks = {
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        }
        
        # Store pose history for smoothing
        self.pose_history: List[PoseFrame] = []
        self.max_history_length = 10
        
    def detect_pose(self, frame: np.ndarray, frame_number: int = 0, timestamp: float = 0.0) -> Optional[PoseFrame]:
        """
        Detect pose in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            frame_number: Frame number for tracking
            timestamp: Timestamp of the frame
            
        Returns:
            PoseFrame object with detected keypoints, or None if no pose detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
            
        # Extract keypoints
        keypoints = {}
        for name, landmark_id in self.key_landmarks.items():
            landmark = results.pose_landmarks.landmark[landmark_id]
            keypoints[name] = PoseKeypoint(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                confidence=landmark.visibility
            )
        
        pose_frame = PoseFrame(
            keypoints=keypoints,
            timestamp=timestamp,
            frame_number=frame_number
        )
        
        # Add to history for smoothing
        self.pose_history.append(pose_frame)
        if len(self.pose_history) > self.max_history_length:
            self.pose_history.pop(0)
            
        return pose_frame
    
    def smooth_pose_data(self, pose_frames: List[PoseFrame], window_length: int = 5) -> List[PoseFrame]:
        """
        Apply smoothing to pose data to reduce jitter.
        
        Args:
            pose_frames: List of pose frames to smooth
            window_length: Length of smoothing window (must be odd)
            
        Returns:
            List of smoothed pose frames
        """
        if len(pose_frames) < window_length:
            return pose_frames
            
        if window_length % 2 == 0:
            window_length += 1  # Ensure odd length
            
        smoothed_frames = []
        
        for i, frame in enumerate(pose_frames):
            # Get window of frames around current frame
            start_idx = max(0, i - window_length // 2)
            end_idx = min(len(pose_frames), i + window_length // 2 + 1)
            window_frames = pose_frames[start_idx:end_idx]
            
            # Smooth each keypoint
            smoothed_keypoints = {}
            for keypoint_name in self.key_landmarks.keys():
                x_values = [f.keypoints[keypoint_name].x for f in window_frames]
                y_values = [f.keypoints[keypoint_name].y for f in window_frames]
                z_values = [f.keypoints[keypoint_name].z for f in window_frames]
                conf_values = [f.keypoints[keypoint_name].confidence for f in window_frames]
                
                # Apply Savitzky-Golay smoothing
                if len(x_values) >= 3:
                    try:
                        smoothed_x = savgol_filter(x_values, min(3, len(x_values)), 1)
                        smoothed_y = savgol_filter(y_values, min(3, len(y_values)), 1)
                        smoothed_z = savgol_filter(z_values, min(3, len(z_values)), 1)
                        
                        # Use median for confidence to avoid over-smoothing
                        smoothed_conf = np.median(conf_values)
                        
                        smoothed_keypoints[keypoint_name] = PoseKeypoint(
                            x=smoothed_x[len(smoothed_x)//2],
                            y=float(smoothed_y[len(smoothed_y)//2]),
                            z=float(smoothed_z[len(smoothed_z)//2]),
                            confidence=float(smoothed_conf)
                        )
                    except:
                        # Fallback to original values if smoothing fails
                        smoothed_keypoints[keypoint_name] = frame.keypoints[keypoint_name]
                else:
                    smoothed_keypoints[keypoint_name] = frame.keypoints[keypoint_name]
            
            smoothed_frames.append(PoseFrame(
                keypoints=smoothed_keypoints,
                timestamp=frame.timestamp,
                frame_number=frame.frame_number
            ))
        
        return smoothed_frames
    
    def calculate_joint_angles(self, pose_frame: PoseFrame) -> Dict[str, float]:
        """
        Calculate joint angles from pose keypoints.
        
        Args:
            pose_frame: Pose frame with keypoints
            
        Returns:
            Dictionary of joint angles in degrees
        """
        angles = {}
        
        # Helper function to calculate angle between three points
        def angle_between_points(p1: PoseKeypoint, p2: PoseKeypoint, p3: PoseKeypoint) -> float:
            """Calculate angle between three points (p2 is the vertex)."""
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
            angle = np.arccos(cos_angle)
            return np.degrees(angle)
        
        # Calculate shoulder angles
        if all(k in pose_frame.keypoints for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            angles['left_shoulder_angle'] = angle_between_points(
                pose_frame.keypoints['left_elbow'],
                pose_frame.keypoints['left_shoulder'],
                pose_frame.keypoints['left_wrist']
            )
        
        if all(k in pose_frame.keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            angles['right_shoulder_angle'] = angle_between_points(
                pose_frame.keypoints['right_elbow'],
                pose_frame.keypoints['right_shoulder'],
                pose_frame.keypoints['right_wrist']
            )
        
        # Calculate elbow angles
        if all(k in pose_frame.keypoints for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
            angles['left_elbow_angle'] = angle_between_points(
                pose_frame.keypoints['left_shoulder'],
                pose_frame.keypoints['left_elbow'],
                pose_frame.keypoints['left_wrist']
            )
        
        if all(k in pose_frame.keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            angles['right_elbow_angle'] = angle_between_points(
                pose_frame.keypoints['right_shoulder'],
                pose_frame.keypoints['right_elbow'],
                pose_frame.keypoints['right_wrist']
            )
        
        # Calculate hip angles
        if all(k in pose_frame.keypoints for k in ['left_hip', 'left_knee', 'left_ankle']):
            angles['left_hip_angle'] = angle_between_points(
                pose_frame.keypoints['left_knee'],
                pose_frame.keypoints['left_hip'],
                pose_frame.keypoints['left_ankle']
            )
        
        if all(k in pose_frame.keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
            angles['right_hip_angle'] = angle_between_points(
                pose_frame.keypoints['right_knee'],
                pose_frame.keypoints['right_hip'],
                pose_frame.keypoints['right_ankle']
            )
        
        # Calculate knee angles
        if all(k in pose_frame.keypoints for k in ['left_hip', 'left_knee', 'left_ankle']):
            angles['left_knee_angle'] = angle_between_points(
                pose_frame.keypoints['left_hip'],
                pose_frame.keypoints['left_knee'],
                pose_frame.keypoints['left_ankle']
            )
        
        if all(k in pose_frame.keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
            angles['right_knee_angle'] = angle_between_points(
                pose_frame.keypoints['right_hip'],
                pose_frame.keypoints['right_knee'],
                pose_frame.keypoints['right_ankle']
            )
        
        return angles
    
    def release(self):
        """Release resources."""
        self.pose.close() 