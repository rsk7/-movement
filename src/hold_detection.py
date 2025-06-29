#!/usr/bin/env python3
"""
Hold detection module for climbing videos.
Detects climbing holds using color-based and shape-based methods.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pose_detection import PoseFrame


@dataclass
class Hold:
    """Represents a detected climbing hold."""
    contour: np.ndarray
    center: Tuple[int, int]
    area: float
    color: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height


class HoldDetector:
    """Detects climbing holds in video frames."""
    
    def __init__(self):
        """Initialize the hold detector."""
        # Define hold color ranges in HSV (adjust based on your gym's holds)
        self.hold_colors = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255]),
            'purple': ([130, 100, 100], [160, 255, 255]),
            'orange': ([10, 100, 100], [20, 255, 255]),
            'pink': ([160, 100, 100], [180, 255, 255])
        }
        
        # Detection parameters
        self.min_hold_area = 100  # Minimum hold area in pixels
        self.max_hold_area = 10000  # Maximum hold area in pixels
        self.min_circularity = 0.3  # Minimum circularity for hold shapes
        self.contact_threshold = 50  # Distance threshold for hold contact (pixels)
    
    def detect_holds(self, frame: np.ndarray) -> List[Hold]:
        """
        Detect climbing holds in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detected holds
        """
        holds = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect holds for each color
        for color_name, (lower, upper) in self.hold_colors.items():
            color_holds = self._detect_holds_by_color(hsv, color_name, lower, upper)
            holds.extend(color_holds)
        
        return holds
    
    def _detect_holds_by_color(self, hsv: np.ndarray, color_name: str, 
                              lower: List[int], upper: List[int]) -> List[Hold]:
        """Detect holds of a specific color."""
        holds = []
        
        # Create color mask
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_hold_area or area > self.max_hold_area:
                continue
            
            # Calculate shape properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Filter by circularity (holds are roughly circular/oval)
            if circularity < self.min_circularity:
                continue
            
            # Calculate hold properties
            center = self._get_contour_center(contour)
            bbox = cv2.boundingRect(contour)
            bbox_tuple = (bbox[0], bbox[1], bbox[2], bbox[3])  # Convert to tuple
            confidence = min(circularity * 2, 1.0)  # Simple confidence based on circularity
            
            hold = Hold(
                contour=contour,
                center=center,
                area=area,
                color=color_name,
                confidence=confidence,
                bbox=bbox_tuple
            )
            
            holds.append(hold)
        
        return holds
    
    def _get_contour_center(self, contour: np.ndarray) -> Tuple[int, int]:
        """Calculate the center of a contour."""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # Fallback to bounding rectangle center
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
        
        return (cx, cy)
    
    def detect_hold_contacts(self, pose_frame: PoseFrame, holds: List[Hold]) -> Dict[str, List[Hold]]:
        """
        Detect which holds are being contacted by the climber.
        
        Args:
            pose_frame: Current pose data
            holds: List of detected holds
            
        Returns:
            Dictionary mapping body parts to contacted holds
        """
        contacts = {
            'left_hand': [],
            'right_hand': [],
            'left_foot': [],
            'right_foot': []
        }
        
        if not pose_frame or not pose_frame.keypoints:
            return contacts
        
        # Body part to keypoint mapping
        body_parts = {
            'left_hand': 'left_wrist',
            'right_hand': 'right_wrist',
            'left_foot': 'left_ankle',
            'right_foot': 'right_ankle'
        }
        
        height, width = (1080, 1920)  # Default frame size, can be overridden
        
        for body_part, keypoint_name in body_parts.items():
            if keypoint_name in pose_frame.keypoints:
                keypoint = pose_frame.keypoints[keypoint_name]
                
                if keypoint.confidence > 0.3:  # Only consider confident detections
                    # Convert normalized coordinates to pixel coordinates
                    px = int(keypoint.x * width)
                    py = int(keypoint.y * height)
                    
                    # Check distance to each hold
                    for hold in holds:
                        distance = np.sqrt((px - hold.center[0])**2 + (py - hold.center[1])**2)
                        
                        if distance < self.contact_threshold:
                            contacts[body_part].append(hold)
        
        return contacts
    
    def draw_holds(self, frame: np.ndarray, holds: List[Hold], 
                   show_contacts: bool = False, contacts: Optional[Dict[str, List[Hold]]] = None) -> np.ndarray:
        """
        Draw detected holds on a frame.
        
        Args:
            frame: Input frame
            holds: List of detected holds
            show_contacts: Whether to highlight contacted holds
            contacts: Dictionary of hold contacts
            
        Returns:
            Frame with holds drawn
        """
        result = frame.copy()
        
        for hold in holds:
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
    
    def get_hold_statistics(self, holds: List[Hold]) -> Dict[str, object]:
        """
        Get statistics about detected holds.
        
        Args:
            holds: List of detected holds
            
        Returns:
            Dictionary with hold statistics
        """
        if not holds:
            return {
                'total_holds': 0,
                'holds_by_color': {},
                'avg_area': 0,
                'avg_confidence': 0
            }
        
        # Count holds by color
        holds_by_color = {}
        for hold in holds:
            holds_by_color[hold.color] = holds_by_color.get(hold.color, 0) + 1
        
        # Calculate averages
        avg_area = np.mean([hold.area for hold in holds])
        avg_confidence = np.mean([hold.confidence for hold in holds])
        
        return {
            'total_holds': len(holds),
            'holds_by_color': holds_by_color,
            'avg_area': avg_area,
            'avg_confidence': avg_confidence
        } 