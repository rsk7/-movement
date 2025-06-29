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
    com: Optional[Tuple[float, float, float]] = None


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
        
        # Define body segments and their relative masses (based on Dempster's data)
        # These are approximate percentages of total body mass
        self.segment_masses = {
            'head': 0.081,      # Head and neck
            'torso': 0.497,     # Trunk (thorax + abdomen)
            'left_upper_arm': 0.028,
            'right_upper_arm': 0.028,
            'left_forearm': 0.016,
            'right_forearm': 0.016,
            'left_hand': 0.006,
            'right_hand': 0.006,
            'left_thigh': 0.100,
            'right_thigh': 0.100,
            'left_shin': 0.046,
            'right_shin': 0.046,
            'left_foot': 0.014,
            'right_foot': 0.014
        }
        
        # Define segment endpoints for COM calculation
        self.segment_definitions = {
            'head': ['nose', 'left_ear', 'right_ear'],
            'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
            'left_upper_arm': ['left_shoulder', 'left_elbow'],
            'right_upper_arm': ['right_shoulder', 'right_elbow'],
            'left_forearm': ['left_elbow', 'left_wrist'],
            'right_forearm': ['right_elbow', 'right_wrist'],
            'left_hand': ['left_wrist', 'left_pinky', 'left_index'],
            'right_hand': ['right_wrist', 'right_pinky', 'right_index'],
            'left_thigh': ['left_hip', 'left_knee'],
            'right_thigh': ['right_hip', 'right_knee'],
            'left_shin': ['left_knee', 'left_ankle'],
            'right_shin': ['right_knee', 'right_ankle'],
            'left_foot': ['left_ankle', 'left_heel', 'left_foot_index'],
            'right_foot': ['right_ankle', 'right_heel', 'right_foot_index']
        }
        
        # COM position within each segment (as fraction from proximal to distal end)
        self.segment_com_fractions = {
            'head': 0.5,        # Center of head
            'torso': 0.5,       # Center of torso
            'left_upper_arm': 0.436,
            'right_upper_arm': 0.436,
            'left_forearm': 0.430,
            'right_forearm': 0.430,
            'left_hand': 0.5,
            'right_hand': 0.5,
            'left_thigh': 0.433,
            'right_thigh': 0.433,
            'left_shin': 0.433,
            'right_shin': 0.433,
            'left_foot': 0.5,
            'right_foot': 0.5
        }
        
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
        
        # Calculate center of mass
        landmarks = self.get_landmarks(results)
        if landmarks:
            pose_frame.com = self.calculate_center_of_mass(landmarks)
        
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
                frame_number=frame.frame_number,
                com=frame.com  # Preserve the COM data
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

    def get_landmarks(self, results):
        """Extract landmark coordinates from pose detection results."""
        if not results.pose_landmarks:
            return None
            
        landmarks = {}
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks[i] = (landmark.x, landmark.y, landmark.z)
        return landmarks

    def calculate_segment_com(self, landmarks: Dict, segment_name: str) -> Optional[Tuple[float, float, float]]:
        """Calculate the center of mass for a specific body segment."""
        if segment_name not in self.segment_definitions:
            return None
            
        segment_points = self.segment_definitions[segment_name]
        valid_points = []
        
        # Map MediaPipe landmark indices to segment points
        landmark_mapping = {
            'nose': 0, 'left_ear': 3, 'right_ear': 8,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
        
        # Collect valid landmark coordinates
        for point_name in segment_points:
            if point_name in landmark_mapping:
                idx = landmark_mapping[point_name]
                if idx in landmarks:
                    valid_points.append(landmarks[idx])
        
        if len(valid_points) < 2:
            return None
            
        # Calculate segment COM based on number of points
        if len(valid_points) == 2:
            # For segments with 2 points (most limbs), use COM fraction
            start_point = np.array(valid_points[0])
            end_point = np.array(valid_points[1])
            com_fraction = self.segment_com_fractions[segment_name]
            com = start_point + com_fraction * (end_point - start_point)
        else:
            # For segments with multiple points (head, torso, hands, feet), use centroid
            com = np.mean(valid_points, axis=0)
            
        return tuple(com)

    def calculate_center_of_mass(self, landmarks: Dict) -> Optional[Tuple[float, float, float]]:
        """Calculate the overall center of mass of the body."""
        if not landmarks:
            return None
            
        total_mass = 0
        weighted_com = np.zeros(3)
        
        # Calculate COM for each segment
        for segment_name, mass_fraction in self.segment_masses.items():
            segment_com = self.calculate_segment_com(landmarks, segment_name)
            if segment_com is not None:
                weighted_com += mass_fraction * np.array(segment_com)
                total_mass += mass_fraction
        
        if total_mass > 0:
            overall_com = weighted_com / total_mass
            return tuple(overall_com)
        
        return None

    def get_pose_data(self, frame):
        """Get pose detection results and calculate COM."""
        results = self.detect_pose(frame)
        landmarks = self.get_landmarks(results)
        
        pose_data = {
            'results': results,
            'landmarks': landmarks,
            'com': None
        }
        
        if landmarks:
            pose_data['com'] = self.calculate_center_of_mass(landmarks)
            
        return pose_data 