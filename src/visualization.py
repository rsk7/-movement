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
        Draw skeleton overlay on frame with enhanced visual styling.
        
        Args:
            frame: Input frame
            pose_frame: Pose data for the frame
            
        Returns:
            Frame with skeleton overlay
        """
        if not pose_frame or not pose_frame.keypoints:
            return frame
            
        height, width = frame.shape[:2]
        
        # Draw skeleton connections with gradient and confidence-based styling
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
                    # Calculate confidence-based color and thickness
                    avg_confidence = (joint1.confidence + joint2.confidence) / 2
                    color_intensity = int(255 * avg_confidence)
                    
                    # Create gradient effect - brighter for higher confidence
                    line_color = (0, color_intensity, 0)  # Green with varying intensity
                    line_thickness = max(1, int(self.line_thickness * avg_confidence))
                    
                    # Draw main line
                    cv2.line(frame, (x1, y1), (x2, y2), line_color, line_thickness)
                    
                    # Add subtle glow effect for high confidence connections
                    if avg_confidence > 0.7:
                        glow_color = (0, min(255, color_intensity + 50), 0)
                        cv2.line(frame, (x1, y1), (x2, y2), glow_color, line_thickness + 1)
        
        # Draw joint markers with enhanced styling
        for joint_name, keypoint in pose_frame.keypoints.items():
            if keypoint.confidence > 0.3:
                x = int(keypoint.x * width)
                y = int(keypoint.y * height)
                
                # Confidence-based joint styling
                confidence = keypoint.confidence
                joint_size = int(self.joint_radius * confidence)
                color_intensity = int(255 * confidence)
                
                # Create gradient joint color (red with varying intensity)
                joint_color = (0, 0, color_intensity)
                
                # Draw outer glow for high confidence joints
                if confidence > 0.7:
                    glow_color = (0, 0, min(255, color_intensity + 30))
                    cv2.circle(frame, (x, y), joint_size + 2, glow_color, -1)
                
                # Draw main joint
                cv2.circle(frame, (x, y), joint_size, joint_color, -1)
                
                # Add white center for contrast
                cv2.circle(frame, (x, y), max(1, joint_size // 2), (255, 255, 255), -1)
        
        return frame
    
    def draw_motion_trails(self, frame: np.ndarray, pose_history: List[PoseFrame], 
                          trail_length: int = 30, fade_factor: float = 0.8) -> np.ndarray:
        """
        Draw motion trails showing recent movement history with smoothing.
        
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
            
            # Draw smoothed trail with fading
            if len(points) > 2:
                # Apply smoothing using spline interpolation
                smoothed_points = self._smooth_trail_points(points)
                
                # Draw smoothed trail
                for i in range(len(smoothed_points) - 1):
                    alpha = (i / len(smoothed_points)) * fade_factor
                    color = tuple(int(c * alpha) for c in self.trail_color)
                    thickness = max(1, self.line_thickness // 2)
                    
                    # Draw line segment
                    cv2.line(frame, smoothed_points[i], smoothed_points[i + 1], 
                            color, thickness)
            elif len(points) == 2:
                # Simple line for just 2 points
                alpha = fade_factor
                color = tuple(int(c * alpha) for c in self.trail_color)
                cv2.line(frame, points[0], points[1], color, max(1, self.line_thickness // 2))
        
        return frame
    
    def _smooth_trail_points(self, points: List[Tuple[int, int]], 
                           smoothing_factor: float = 0.3) -> List[Tuple[int, int]]:
        """
        Smooth trail points using simple moving average.
        
        Args:
            points: List of (x, y) coordinate tuples
            smoothing_factor: Smoothing factor (0-1, higher = smoother)
            
        Returns:
            List of smoothed (x, y) coordinate tuples
        """
        if len(points) < 3:
            return points
        
        # Use simple moving average smoothing
        return self._simple_smooth_points(points)
    
    def _simple_smooth_points(self, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Simple smoothing using moving average.
        
        Args:
            points: List of (x, y) coordinate tuples
            
        Returns:
            List of smoothed (x, y) coordinate tuples
        """
        if len(points) < 3:
            return points
        
        smoothed = [points[0]]  # Keep first point
        
        # Apply moving average smoothing
        for i in range(1, len(points) - 1):
            x_avg = (points[i-1][0] + points[i][0] + points[i+1][0]) // 3
            y_avg = (points[i-1][1] + points[i][1] + points[i+1][1]) // 3
            smoothed.append((x_avg, y_avg))
        
        smoothed.append(points[-1])  # Keep last point
        return smoothed
    
    def draw_joint_angles(self, frame: np.ndarray, pose_frame: PoseFrame, 
                         angles: Dict[str, float]) -> np.ndarray:
        """
        Draw joint angles on frame, positioned away from joints to avoid obscuring skeleton.
        
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
        
        # Map angle keys to joint names and offset directions
        angle_to_joint_offset = {
            'left_shoulder_angle': ('left_shoulder', (-50, -30)),  # Up and left
            'right_shoulder_angle': ('right_shoulder', (50, -30)),  # Up and right
            'left_elbow_angle': ('left_elbow', (-40, 0)),  # Left
            'right_elbow_angle': ('right_elbow', (40, 0)),  # Right
            'left_hip_angle': ('left_hip', (-50, 30)),  # Down and left
            'right_hip_angle': ('right_hip', (50, 30)),  # Down and right
            'left_knee_angle': ('left_knee', (-40, 20)),  # Left and down
            'right_knee_angle': ('right_knee', (40, 20)),  # Right and down
        }
        
        # Draw angles for key joints
        for angle_name, angle_value in angles.items():
            if angle_name in angle_to_joint_offset:
                joint_name, (offset_x, offset_y) = angle_to_joint_offset[angle_name]
                if joint_name in pose_frame.keypoints:
                    keypoint = pose_frame.keypoints[joint_name]
                    if keypoint.confidence > 0.3:
                        # Base position at joint
                        base_x = int(keypoint.x * width)
                        base_y = int(keypoint.y * height)
                        
                        # Calculate offset position for text
                        text_x = base_x + offset_x
                        text_y = base_y + offset_y
                        
                        # Ensure text stays within frame bounds
                        text_x = max(20, min(width - 20, text_x))
                        text_y = max(20, min(height - 20, text_y))
                        
                        # Draw angle text
                        text = f"{angle_value:.1f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.4
                        thickness = 1
                        
                        # Get text size for background
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        # Draw background rectangle with rounded corners effect
                        padding = 3
                        cv2.rectangle(frame, 
                                    (text_x - text_width//2 - padding, text_y - text_height - padding),
                                    (text_x + text_width//2 + padding, text_y + baseline + padding),
                                    (0, 0, 0), -1)
                        
                        # Draw text
                        cv2.putText(frame, text, 
                                  (text_x - text_width//2, text_y), 
                                  font, font_scale, self.text_color, thickness)
                        
                        # Draw a small line connecting joint to text (optional)
                        line_color = (100, 100, 100)  # Gray line
                        cv2.line(frame, (base_x, base_y), (text_x, text_y), line_color, 1)
        
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
                           show_com: bool = True,
                           trail_length: int = 30,
                           output_width: Optional[int] = None,
                           output_height: Optional[int] = None,
                           holds: Optional[List] = None,
                           hold_contacts: Optional[Dict] = None,
                           show_holds: bool = False,
                           show_hold_contacts: bool = False,
                           show_rhythm: bool = False,
                           show_motion_blur: bool = False,
                           show_energy: bool = False) -> np.ndarray:
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
            show_com: Whether to show center of mass
            trail_length: Length of motion trails
            output_width: Target output width for landmark scaling
            output_height: Target output height for landmark scaling
            holds: List of detected holds
            hold_contacts: Dictionary of hold contacts
            show_holds: Whether to show holds
            show_hold_contacts: Whether to show hold contacts
            show_rhythm: Whether to show rhythm information
            show_motion_blur: Whether to show motion blur effect
            show_energy: Whether to show energy visualization
            
        Returns:
            Frame with all overlays applied
        """
        # Make a copy to avoid modifying original
        overlay_frame = frame.copy()
        
        # Use output dimensions if provided, otherwise use frame dimensions
        draw_width = output_width if output_width is not None else frame.shape[1]
        draw_height = output_height if output_height is not None else frame.shape[0]
        
        # Resize frame to output dimensions if needed
        if output_width is not None and output_height is not None:
            if frame.shape[1] != output_width or frame.shape[0] != output_height:
                overlay_frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
        
        # Draw holds first (behind everything else)
        if show_holds and holds:
            overlay_frame = self.draw_holds(overlay_frame, holds, show_hold_contacts, hold_contacts)
        
        # Draw motion trails first (behind skeleton)
        if show_trails and pose_history:
            overlay_frame = self.draw_motion_trails(overlay_frame, pose_history, trail_length)
        
        # Draw COM trail (behind skeleton)
        if show_com and pose_history:
            overlay_frame = self.draw_com_trail(overlay_frame, pose_history, draw_width, draw_height, trail_length)
        
        # Draw skeleton
        if pose_frame:
            overlay_frame = self.draw_pose_skeleton(overlay_frame, pose_frame)
        
        # Draw COM position (on top of skeleton)
        if show_com and pose_frame and pose_frame.com:
            # Get movement type from rhythm events if available
            movement_type = None
            rhythm_events = getattr(pose_frame, 'rhythm_events', [])
            if rhythm_events:
                latest_event = rhythm_events[-1]
                movement_type = latest_event.event_type
            
            overlay_frame = self.draw_center_of_mass(overlay_frame, pose_frame.com, draw_width, draw_height, 
                                                   movement_type=movement_type)
        
        # Draw joint angles
        if show_angles and angles:
            overlay_frame = self.draw_joint_angles(overlay_frame, pose_frame, angles)
        
        # Draw velocity vectors
        if show_velocity and pose_history:
            overlay_frame = self.draw_velocity_vectors(overlay_frame, pose_history)
        
        # Draw rhythm information
        if show_rhythm:
            # Get rhythm summary from the rhythm detector (passed as parameter)
            rhythm_summary = getattr(pose_frame, 'rhythm_summary', {})
            rhythm_events = getattr(pose_frame, 'rhythm_events', [])
            
            overlay_frame = self.draw_rhythm_info(overlay_frame, rhythm_summary, draw_width, draw_height)
            overlay_frame = self.draw_rhythm_events(overlay_frame, rhythm_events, draw_width, draw_height)
            overlay_frame = self.draw_movement_classification(overlay_frame, rhythm_events, draw_width, draw_height)
        
        # Draw motion blur effect
        if show_motion_blur and pose_history:
            overlay_frame = self.draw_motion_blur(overlay_frame, pose_history)
        
        # Draw energy visualization
        if show_energy and pose_history:
            overlay_frame = self.draw_energy_visualization(overlay_frame, pose_history)
        
        return overlay_frame
    
    def create_overlay_only_frame(self, width: int, height: int, pose_frame: PoseFrame,
                                 pose_history: Optional[List[PoseFrame]] = None,
                                 angles: Optional[Dict[str, float]] = None,
                                 show_trails: bool = True,
                                 show_angles: bool = True,
                                 show_velocity: bool = False,
                                 show_com: bool = True,
                                 show_rhythm: bool = False,
                                 trail_length: int = 30) -> np.ndarray:
        """
        Create a frame with only pose overlays (no background video).
        
        Args:
            width: Frame width
            height: Frame height
            pose_frame: Current pose data
            pose_history: History of pose frames for trails
            angles: Joint angles to display
            show_trails: Whether to show motion trails
            show_angles: Whether to show joint angles
            show_velocity: Whether to show velocity vectors
            show_com: Whether to show center of mass
            show_rhythm: Whether to show rhythm information
            trail_length: Length of motion trails
            
        Returns:
            Frame with only pose overlays on transparent/black background
        """
        # Create a black background frame
        overlay_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw motion trails first (behind skeleton)
        if show_trails and pose_history:
            overlay_frame = self.draw_motion_trails(overlay_frame, pose_history, trail_length)
        
        # Draw COM trail (behind skeleton)
        if show_com and pose_history:
            overlay_frame = self.draw_com_trail(overlay_frame, pose_history, width, height, trail_length)
        
        # Draw skeleton
        overlay_frame = self.draw_pose_skeleton(overlay_frame, pose_frame)
        
        # Draw COM position (on top of skeleton)
        if show_com and pose_frame and pose_frame.com:
            # Get movement type from rhythm events if available
            movement_type = None
            rhythm_events = getattr(pose_frame, 'rhythm_events', [])
            if rhythm_events:
                latest_event = rhythm_events[-1]
                movement_type = latest_event.event_type
            
            overlay_frame = self.draw_center_of_mass(overlay_frame, pose_frame.com, width, height, 
                                                   movement_type=movement_type)
        
        # Draw joint angles
        if show_angles and angles:
            overlay_frame = self.draw_joint_angles(overlay_frame, pose_frame, angles)
        
        # Draw velocity vectors
        if show_velocity and pose_history:
            overlay_frame = self.draw_velocity_vectors(overlay_frame, pose_history)
        
        # Draw rhythm information
        if show_rhythm:
            # Get rhythm summary from the rhythm detector (passed as parameter)
            rhythm_summary = getattr(pose_frame, 'rhythm_summary', {})
            overlay_frame = self.draw_rhythm_info(overlay_frame, rhythm_summary, width, height)
            
            # Draw rhythm events
            rhythm_events = getattr(pose_frame, 'rhythm_events', [])
            overlay_frame = self.draw_rhythm_events(overlay_frame, rhythm_events, width, height)
            overlay_frame = self.draw_movement_classification(overlay_frame, rhythm_events, width, height)
        
        return overlay_frame
    
    def draw_holds(self, frame: np.ndarray, holds: List, 
                   show_contacts: bool = False, contacts: Optional[Dict] = None) -> np.ndarray:
        """
        Draw holds on a frame.
        
        Args:
            frame: Input frame
            holds: List of detected holds
            show_contacts: Whether to highlight contacted holds
            contacts: Dictionary of hold contacts
            
        Returns:
            Frame with holds drawn
        """
        result = frame.copy()
        
        for i, hold in enumerate(holds):
            # Choose color based on hold color
            color_map = {
                'red': (0, 0, 255),
                'blue': (255, 0, 0),
                'yellow': (0, 255, 255),
                'green': (0, 255, 0),
                'purple': (255, 0, 255),
                'orange': (0, 165, 255),
                'pink': (147, 20, 255)
            }
            
            color = color_map.get(hold.color, (255, 255, 255))
            
            # Check if this hold is being contacted
            is_contacted = False
            if show_contacts and contacts:
                for body_part_holds in contacts.values():
                    for h in body_part_holds:
                        if tuple(map(int, hold.center)) == tuple(map(int, h.center)) and hold.color == h.color:
                            is_contacted = True
                            break
                    if is_contacted:
                        break
            
            # Draw hold contour
            thickness = 3 if is_contacted else 2
            cv2.drawContours(result, [hold.contour], -1, color, thickness)
            
            # Draw hold center
            cv2.circle(result, hold.center, 5, color, -1)
            
            # Draw hold info
            text = f"{hold.color} ({hold.confidence:.2f})"
            cv2.putText(result, text, 
                       (hold.center[0] - 20, hold.center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Highlight contacted holds
            if is_contacted:
                cv2.circle(result, hold.center, 15, (255, 255, 255), 2)
        
        return result
    
    def draw_center_of_mass(self, frame: np.ndarray, com: Tuple[float, float, float], 
                           frame_width: int, frame_height: int, 
                           color: Tuple[int, int, int] = (0, 255, 255),
                           movement_type: Optional[str] = None) -> np.ndarray:
        """
        Draw the center of mass on the frame with optional movement type label.
        
        Args:
            frame: Input frame
            com: Center of mass coordinates (x, y, z) in normalized coordinates
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            color: Color for COM visualization (BGR format)
            movement_type: Optional movement type to display next to COM
            
        Returns:
            Frame with COM visualization
        """
        if com is None:
            return frame
            
        # Convert normalized coordinates to pixel coordinates
        x = int(com[0] * frame_width)
        y = int(com[1] * frame_height)
        
        # Draw COM as a filled circle with a cross
        cv2.circle(frame, (x, y), 8, color, -1)  # Filled circle
        cv2.circle(frame, (x, y), 12, color, 2)  # Outline circle
        
        # Draw cross
        cv2.line(frame, (x-6, y), (x+6, y), (255, 255, 255), 2)
        cv2.line(frame, (x, y-6), (x, y+6), (255, 255, 255), 2)
        
        # Add "COM" label
        cv2.putText(frame, "COM", (x+15, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 1)
        
        # Add movement type label if provided
        if movement_type:
            # Get color for movement type
            movement_color_map = {
                'reach': (0, 0, 255),    # Red
                'pull': (0, 165, 255),   # Orange
                'step': (255, 0, 255),   # Purple
                'pause': (0, 255, 0),    # Green
                'steady': (255, 0, 0),   # Blue
            }
            movement_color = movement_color_map.get(movement_type, (255, 255, 255))
            
            # Add background rectangle for better visibility
            label_text = movement_type.upper()
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Position label below COM
            label_x = x - text_width // 2
            label_y = y + 25
            
            # Draw background rectangle
            cv2.rectangle(frame, (label_x - 5, label_y - text_height - 5), 
                         (label_x + text_width + 5, label_y + 5), (0, 0, 0), -1)
            cv2.rectangle(frame, (label_x - 5, label_y - text_height - 5), 
                         (label_x + text_width + 5, label_y + 5), movement_color, 2)
            
            # Draw movement type text
            cv2.putText(frame, label_text, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, movement_color, 2)
        
        return frame

    def draw_com_trail(self, frame: np.ndarray, pose_history: List[PoseFrame], 
                      frame_width: int, frame_height: int,
                      trail_length: int = 30, 
                      color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        Draw a trail of center of mass positions with smoothing.
        
        Args:
            frame: Input frame
            pose_history: List of pose frames with COM data
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            trail_length: Number of previous COM positions to show
            color: Color for COM trail (BGR format)
            
        Returns:
            Frame with COM trail visualization
        """
        if not pose_history:
            return frame
            
        # Get recent COM positions (assuming COM is stored in pose_frame attributes)
        com_positions = []
        for pose_frame in reversed(pose_history[-trail_length:]):
            if pose_frame.com is not None:
                com = pose_frame.com
                x = int(com[0] * frame_width)
                y = int(com[1] * frame_height)
                com_positions.append((x, y))
        
        # Draw smoothed trail
        if len(com_positions) > 2:
            # Apply smoothing
            smoothed_points = self._simple_smooth_points(com_positions)
            
            # Draw smoothed trail with fading
            for i in range(len(smoothed_points) - 1):
                # Fade color based on age
                alpha = 1.0 - (i / len(smoothed_points))
                trail_color = tuple(int(c * alpha) for c in color)
                
                cv2.line(frame, smoothed_points[i], smoothed_points[i+1], 
                        trail_color, 2)
        elif len(com_positions) == 2:
            # Simple line for just 2 points
            alpha = 0.5
            trail_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, com_positions[0], com_positions[1], 
                    trail_color, 2)
        
        return frame
    
    def draw_rhythm_info(self, frame: np.ndarray, rhythm_summary: Dict, 
                        frame_width: int, frame_height: int) -> np.ndarray:
        """
        Draw rhythm information on the frame.
        
        Args:
            frame: Input frame
            rhythm_summary: Rhythm analysis summary
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Frame with rhythm information overlay
        """
        if rhythm_summary.get("status") == "insufficient_data":
            return frame
        
        # Get rhythm metrics
        rhythm_score = rhythm_summary.get("average_rhythm_score", 0)
        pattern_type = rhythm_summary.get("current_pattern", "unknown")
        tempo = rhythm_summary.get("average_tempo", 0)
        regularity = rhythm_summary.get("average_regularity", 0)
        
        # Position for rhythm info (top-right corner)
        x_offset = frame_width - 200
        y_offset = 30
        line_height = 25
        
        # Background rectangle for text
        cv2.rectangle(frame, (x_offset - 10, y_offset - 25), 
                     (x_offset + 190, y_offset + 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (x_offset - 10, y_offset - 25), 
                     (x_offset + 190, y_offset + 100), (255, 255, 255), 2)
        
        # Rhythm score with color coding
        score_color = self._get_rhythm_score_color(rhythm_score)
        score_text = f"Rhythm: {rhythm_score:.1%}"
        cv2.putText(frame, score_text, (x_offset, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        
        # Pattern type
        pattern_color = self._get_pattern_color(pattern_type)
        pattern_text = f"Pattern: {pattern_type.upper()}"
        cv2.putText(frame, pattern_text, (x_offset, y_offset + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, pattern_color, 2)
        
        # Tempo
        tempo_text = f"Tempo: {tempo:.1f}/s"
        cv2.putText(frame, tempo_text, (x_offset, y_offset + 2 * line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Regularity
        reg_text = f"Regularity: {regularity:.1%}"
        cv2.putText(frame, reg_text, (x_offset, y_offset + 3 * line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _get_rhythm_score_color(self, score: float) -> Tuple[int, int, int]:
        """Get color for rhythm score (green=good, yellow=medium, red=poor)."""
        if score >= 0.7:
            return (0, 255, 0)  # Green
        elif score >= 0.4:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 0, 255)  # Red
    
    def _get_pattern_color(self, pattern_type: str) -> Tuple[int, int, int]:
        """Get color for pattern type."""
        color_map = {
            'flowing': (0, 255, 0),    # Green
            'steady': (0, 255, 255),   # Yellow
            'burst': (0, 0, 255),      # Red
            'hesitant': (0, 165, 255), # Orange
            'mixed': (255, 255, 255),  # White
            'unknown': (128, 128, 128) # Gray
        }
        return color_map.get(pattern_type, (255, 255, 255))
    
    def draw_rhythm_events(self, frame: np.ndarray, rhythm_events: List, 
                          frame_width: int, frame_height: int) -> np.ndarray:
        """
        Draw rhythm events on the frame.
        
        Args:
            frame: Input frame
            rhythm_events: List of recent rhythm events
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Frame with rhythm event markers
        """
        # Show only recent events (last 10)
        recent_events = rhythm_events[-10:] if len(rhythm_events) > 10 else rhythm_events
        
        for event in recent_events:
            # Convert normalized coordinates to pixel coordinates
            x = int(event.com_position[0] * frame_width)
            y = int(event.com_position[1] * frame_height)
            
            # Get color based on event type
            color_map = {
                'reach': (0, 0, 255),    # Red
                'pull': (0, 165, 255),   # Orange
                'step': (255, 0, 255),   # Purple
                'pause': (0, 255, 0),    # Green
                'steady': (255, 0, 0),   # Blue
            }
            color = color_map.get(event.event_type, (255, 255, 255))
            
            # Draw event marker
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.circle(frame, (x, y), 10, color, 2)
            
            # Add event type label
            label = event.event_type.upper()
            cv2.putText(frame, label, (x + 15, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def draw_movement_classification(self, frame: np.ndarray, rhythm_events: List, 
                                   frame_width: int, frame_height: int) -> np.ndarray:
        """
        Draw movement classification information on the frame.
        
        Args:
            frame: Input frame
            rhythm_events: List of recent rhythm events with movement classification
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Frame with movement classification overlay
        """
        if not rhythm_events:
            return frame
        
        # Get the most recent event
        latest_event = rhythm_events[-1]
        
        # Position for movement info (bottom-left corner)
        x_offset = 20
        y_offset = frame_height - 120
        line_height = 25
        
        # Background rectangle for text
        cv2.rectangle(frame, (x_offset - 10, y_offset - 25), 
                     (x_offset + 200, y_offset + 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (x_offset - 10, y_offset - 25), 
                     (x_offset + 200, y_offset + 100), (255, 255, 255), 2)
        
        # Movement type with color coding
        color_map = {
            'reach': (0, 0, 255),    # Red
            'pull': (0, 165, 255),   # Orange
            'step': (255, 0, 255),   # Purple
            'pause': (0, 255, 0),    # Green
            'steady': (255, 0, 0),   # Blue
        }
        movement_color = color_map.get(latest_event.event_type, (255, 255, 255))
        
        movement_text = f"Movement: {latest_event.event_type.upper()}"
        cv2.putText(frame, movement_text, (x_offset, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, movement_color, 2)
        
        # Confidence
        confidence_text = f"Confidence: {latest_event.confidence:.1%}"
        cv2.putText(frame, confidence_text, (x_offset, y_offset + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Velocity
        velocity_text = f"Velocity: {latest_event.velocity:.3f}"
        cv2.putText(frame, velocity_text, (x_offset, y_offset + 2 * line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Description
        if hasattr(latest_event, 'description') and latest_event.description:
            desc_text = latest_event.description[:30] + "..." if len(latest_event.description) > 30 else latest_event.description
            cv2.putText(frame, desc_text, (x_offset, y_offset + 3 * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def draw_motion_blur(self, frame: np.ndarray, pose_history: List[PoseFrame], 
                        blur_length: int = 5) -> np.ndarray:
        """
        Draw motion blur effect for moving joints.
        
        Args:
            frame: Input frame
            pose_history: List of recent pose frames
            blur_length: Number of frames to use for blur effect
            
        Returns:
            Frame with motion blur effect
        """
        if not pose_history or len(pose_history) < 2:
            return frame
            
        height, width = frame.shape[:2]
        
        # Get recent frames for blur effect
        recent_frames = pose_history[-blur_length:]
        
        # Draw motion blur for key joints
        key_joints = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle', 'nose']
        
        for joint_name in key_joints:
            points = []
            
            # Collect points for this joint
            for pose_frame in recent_frames:
                if joint_name in pose_frame.keypoints:
                    keypoint = pose_frame.keypoints[joint_name]
                    if keypoint.confidence > 0.3:
                        x = int(keypoint.x * width)
                        y = int(keypoint.y * height)
                        points.append((x, y, keypoint.confidence))
            
            # Draw motion blur trail
            if len(points) > 1:
                for i in range(len(points) - 1):
                    # Calculate alpha based on position in trail
                    alpha = i / len(points)
                    
                    # Create blur color (cyan with fading)
                    blur_color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
                    blur_thickness = max(1, int(3 * alpha))
                    
                    # Draw blur line
                    cv2.line(frame, (points[i][0], points[i][1]), 
                            (points[i+1][0], points[i+1][1]), 
                            blur_color, blur_thickness)
        
        return frame
    
    def draw_energy_visualization(self, frame: np.ndarray, pose_history: List[PoseFrame]) -> np.ndarray:
        """
        Draw energy/force visualization showing movement intensity.
        
        Args:
            frame: Input frame
            pose_history: List of recent pose frames
            
        Returns:
            Frame with energy visualization
        """
        if not pose_history or len(pose_history) < 3:
            return frame
            
        height, width = frame.shape[:2]
        
        # Calculate movement energy for key joints
        key_joints = ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle']
        
        for joint_name in key_joints:
            if len(pose_history) >= 3:
                current = pose_history[-1]
                previous = pose_history[-2]
                
                if (joint_name in current.keypoints and 
                    joint_name in previous.keypoints):
                    
                    curr_kp = current.keypoints[joint_name]
                    prev_kp = previous.keypoints[joint_name]
                    
                    if curr_kp.confidence > 0.3 and prev_kp.confidence > 0.3:
                        # Calculate movement velocity
                        dx = (curr_kp.x - prev_kp.x) * width
                        dy = (curr_kp.y - prev_kp.y) * height
                        velocity = np.sqrt(dx**2 + dy**2)
                        
                        # Lower threshold for more visible particles
                        if velocity > 2:  # Reduced threshold from 5 to 2
                            x = int(curr_kp.x * width)
                            y = int(curr_kp.y * height)
                            
                            # Energy intensity based on velocity
                            energy_intensity = min(255, int(velocity * 20))  # Increased multiplier
                            
                            # Draw energy particles with better visibility
                            particle_color = (255, energy_intensity, 0)  # Bright orange/red
                            
                            # Draw multiple particles in a smaller radius for better visibility
                            num_particles = min(10, int(velocity / 1.5))  # More particles, smaller radius
                            for _ in range(num_particles):
                                offset_x = np.random.randint(-3, 4)  # Smaller radius
                                offset_y = np.random.randint(-3, 4)  # Smaller radius
                                particle_x = x + offset_x
                                particle_y = y + offset_y
                                
                                # Ensure particles stay within frame
                                particle_x = max(0, min(width - 1, particle_x))
                                particle_y = max(0, min(height - 1, particle_y))
                                
                                # Draw larger particles for better visibility
                                particle_size = max(2, min(6, int(velocity / 2)))  # Bigger particles (2-6 pixels)
                                cv2.circle(frame, (particle_x, particle_y), particle_size, particle_color, -1)
                                
                                # Add white center for contrast
                                cv2.circle(frame, (particle_x, particle_y), max(1, particle_size // 2), (255, 255, 255), -1)
        
        return frame 