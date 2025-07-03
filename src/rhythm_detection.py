#!/usr/bin/env python3
"""
Rhythm detection module for climbing movement analysis.
Analyzes timing patterns, movement sequences, and rhythm in climbing.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from src.pose_detection import PoseFrame, PoseKeypoint


@dataclass
class MovementFeatures:
    """Features extracted from pose data for movement classification."""
    com_velocity: float
    com_direction: Tuple[float, float]  # (x, y) direction vector
    arm_movement: Dict[str, float]  # Left/right arm velocities
    leg_movement: Dict[str, float]  # Left/right leg velocities
    shoulder_angles: Dict[str, float]  # Left/right shoulder angles
    hip_angles: Dict[str, float]  # Left/right hip angles
    arm_elevation: Dict[str, float]  # How high arms are raised
    leg_elevation: Dict[str, float]  # How high legs are raised


@dataclass
class RhythmEvent:
    """Represents a rhythm event in climbing movement."""
    timestamp: float
    event_type: str  # 'reach', 'pull', 'step', 'release', 'pause'
    confidence: float
    com_position: Tuple[float, float, float]
    velocity: float
    description: str
    movement_features: Optional[MovementFeatures] = None


@dataclass
class RhythmPattern:
    """Represents a detected rhythm pattern."""
    pattern_type: str  # 'steady', 'burst', 'hesitant', 'flowing'
    start_time: float
    end_time: float
    events: List[RhythmEvent]
    rhythm_score: float  # 0-1, higher is more rhythmic
    tempo: float  # events per second
    regularity: float  # consistency of timing


class RhythmDetector:
    """Detects rhythm patterns in climbing movement using COM and skeleton data."""
    
    def __init__(self, 
                 min_velocity_threshold: float = 0.01,
                 max_velocity_threshold: float = 0.1,
                 min_event_interval: float = 0.2,
                 rhythm_window_size: int = 30):
        """
        Initialize the rhythm detector.
        
        Args:
            min_velocity_threshold: Minimum COM velocity to consider as movement
            max_velocity_threshold: Maximum COM velocity for normal movement
            min_event_interval: Minimum time between rhythm events
            rhythm_window_size: Number of frames to analyze for rhythm patterns
        """
        self.min_velocity_threshold = min_velocity_threshold
        self.max_velocity_threshold = max_velocity_threshold
        self.min_event_interval = min_event_interval
        self.rhythm_window_size = rhythm_window_size
        
        # Store movement history
        self.com_history: List[Tuple[float, Tuple[float, float, float]]] = []
        self.velocity_history: List[Tuple[float, float]] = []
        self.pose_history: List[Tuple[float, PoseFrame]] = []
        self.rhythm_events: List[RhythmEvent] = []
        self.rhythm_patterns: List[RhythmPattern] = []
        
        # Movement classification thresholds
        self.reach_velocity_threshold = 0.08
        self.pull_velocity_threshold = 0.06
        self.step_velocity_threshold = 0.05
        self.arm_elevation_threshold = 0.3  # Arms above shoulders
        self.leg_elevation_threshold = 0.2  # Legs above hips
    
    def add_frame(self, timestamp: float, pose_frame: PoseFrame):
        """Add a new frame for rhythm analysis."""
        if pose_frame.com is None:
            return
            
        # Store COM position and timestamp
        self.com_history.append((timestamp, pose_frame.com))
        self.pose_history.append((timestamp, pose_frame))
        
        # Calculate COM velocity
        if len(self.com_history) >= 2:
            prev_time, prev_com = self.com_history[-2]
            curr_time, curr_com = self.com_history[-1]
            
            dt = curr_time - prev_time
            if dt > 0:
                # Calculate 2D velocity (x, y components)
                dx = curr_com[0] - prev_com[0]
                dy = curr_com[1] - prev_com[1]
                velocity = np.sqrt(dx**2 + dy**2) / dt
                
                self.velocity_history.append((timestamp, velocity))
                
                # Extract movement features
                movement_features = self._extract_movement_features(timestamp, pose_frame)
                
                # Detect rhythm events based on velocity changes and movement patterns
                self._detect_rhythm_events(timestamp, velocity, pose_frame.com, movement_features)
        
        # Analyze rhythm patterns periodically
        if len(self.velocity_history) >= self.rhythm_window_size:
            self._analyze_rhythm_patterns()
    
    def _extract_movement_features(self, timestamp: float, pose_frame: PoseFrame) -> MovementFeatures:
        """Extract movement features from pose data for classification."""
        keypoints = pose_frame.keypoints
        
        # Calculate COM velocity and direction
        com_velocity = 0.0
        com_direction = (0.0, 0.0)
        
        if len(self.com_history) >= 2:
            prev_time, prev_com = self.com_history[-2]
            curr_time, curr_com = self.com_history[-1]
            dt = curr_time - prev_time
            if dt > 0:
                dx = curr_com[0] - prev_com[0]
                dy = curr_com[1] - prev_com[1]
                com_velocity = np.sqrt(dx**2 + dy**2) / dt
                if com_velocity > 0:
                    com_direction = (dx / com_velocity, dy / com_velocity)
        
        # Calculate arm and leg movements
        arm_movement = {'left': 0.0, 'right': 0.0}
        leg_movement = {'left': 0.0, 'right': 0.0}
        
        if len(self.pose_history) >= 2:
            prev_time, prev_pose = self.pose_history[-2]
            curr_time, curr_pose = self.pose_history[-1]
            dt = curr_time - prev_time
            if dt > 0:
                # Arm movement (wrist velocity)
                for side in ['left', 'right']:
                    wrist_key = f'{side}_wrist'
                    if wrist_key in prev_pose.keypoints and wrist_key in curr_pose.keypoints:
                        prev_wrist = prev_pose.keypoints[wrist_key]
                        curr_wrist = curr_pose.keypoints[wrist_key]
                        dx = curr_wrist.x - prev_wrist.x
                        dy = curr_wrist.y - prev_wrist.y
                        arm_movement[side] = np.sqrt(dx**2 + dy**2) / dt
                
                # Leg movement (ankle velocity)
                for side in ['left', 'right']:
                    ankle_key = f'{side}_ankle'
                    if ankle_key in prev_pose.keypoints and ankle_key in curr_pose.keypoints:
                        prev_ankle = prev_pose.keypoints[ankle_key]
                        curr_ankle = curr_pose.keypoints[ankle_key]
                        dx = curr_ankle.x - prev_ankle.x
                        dy = curr_ankle.y - prev_ankle.y
                        leg_movement[side] = np.sqrt(dx**2 + dy**2) / dt
        
        # Calculate joint angles
        shoulder_angles = {'left': 0.0, 'right': 0.0}
        hip_angles = {'left': 0.0, 'right': 0.0}
        
        # Calculate shoulder angles (shoulder-elbow-wrist)
        for side in ['left', 'right']:
            shoulder_key = f'{side}_shoulder'
            elbow_key = f'{side}_elbow'
            wrist_key = f'{side}_wrist'
            
            if all(key in keypoints for key in [shoulder_key, elbow_key, wrist_key]):
                shoulder = keypoints[shoulder_key]
                elbow = keypoints[elbow_key]
                wrist = keypoints[wrist_key]
                shoulder_angles[side] = self._calculate_angle(shoulder, elbow, wrist)
        
        # Calculate hip angles (hip-knee-ankle)
        for side in ['left', 'right']:
            hip_key = f'{side}_hip'
            knee_key = f'{side}_knee'
            ankle_key = f'{side}_ankle'
            
            if all(key in keypoints for key in [hip_key, knee_key, ankle_key]):
                hip = keypoints[hip_key]
                knee = keypoints[knee_key]
                ankle = keypoints[ankle_key]
                hip_angles[side] = self._calculate_angle(hip, knee, ankle)
        
        # Calculate arm and leg elevation
        arm_elevation = {'left': 0.0, 'right': 0.0}
        leg_elevation = {'left': 0.0, 'right': 0.0}
        
        # Arm elevation relative to shoulders
        if 'left_shoulder' in keypoints and 'left_wrist' in keypoints:
            shoulder_y = keypoints['left_shoulder'].y
            wrist_y = keypoints['left_wrist'].y
            arm_elevation['left'] = max(0, shoulder_y - wrist_y)  # Higher wrist = lower value
        
        if 'right_shoulder' in keypoints and 'right_wrist' in keypoints:
            shoulder_y = keypoints['right_shoulder'].y
            wrist_y = keypoints['right_wrist'].y
            arm_elevation['right'] = max(0, shoulder_y - wrist_y)
        
        # Leg elevation relative to hips
        if 'left_hip' in keypoints and 'left_ankle' in keypoints:
            hip_y = keypoints['left_hip'].y
            ankle_y = keypoints['left_ankle'].y
            leg_elevation['left'] = max(0, hip_y - ankle_y)
        
        if 'right_hip' in keypoints and 'right_ankle' in keypoints:
            hip_y = keypoints['right_hip'].y
            ankle_y = keypoints['right_ankle'].y
            leg_elevation['right'] = max(0, hip_y - ankle_y)
        
        return MovementFeatures(
            com_velocity=com_velocity,
            com_direction=com_direction,
            arm_movement=arm_movement,
            leg_movement=leg_movement,
            shoulder_angles=shoulder_angles,
            hip_angles=hip_angles,
            arm_elevation=arm_elevation,
            leg_elevation=leg_elevation
        )
    
    def _calculate_angle(self, p1: PoseKeypoint, p2: PoseKeypoint, p3: PoseKeypoint) -> float:
        """Calculate angle between three points (p2 is the vertex)."""
        # Vector from p2 to p1
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        # Vector from p2 to p3
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors
        angle = np.arccos(cos_angle)
        
        return float(np.degrees(angle))
    
    def _classify_movement_type(self, velocity: float, features: MovementFeatures) -> Tuple[str, float, str]:
        """Classify movement type based on velocity and movement features."""
        
        # Calculate feature scores
        arm_movement_score = max(features.arm_movement.values())
        leg_movement_score = max(features.leg_movement.values())
        arm_elevation_score = max(features.arm_elevation.values())
        leg_elevation_score = max(features.leg_elevation.values())
        
        # Upward COM movement (positive y direction)
        upward_movement = features.com_direction[1] > 0.3
        
        # Reach movement: high arm movement, arms elevated, moderate velocity
        reach_score = 0.0
        if velocity > self.reach_velocity_threshold:
            reach_score += 0.4
        if arm_movement_score > 0.05:
            reach_score += 0.3
        if arm_elevation_score > self.arm_elevation_threshold:
            reach_score += 0.3
        
        # Pull movement: upward COM movement, high velocity, arms engaged
        pull_score = 0.0
        if velocity > self.pull_velocity_threshold:
            pull_score += 0.4
        if upward_movement:
            pull_score += 0.4
        if arm_movement_score > 0.03:
            pull_score += 0.2
        
        # Step movement: high leg movement, legs elevated, lateral movement
        step_score = 0.0
        if velocity > self.step_velocity_threshold:
            step_score += 0.3
        if leg_movement_score > 0.04:
            step_score += 0.4
        if leg_elevation_score > self.leg_elevation_threshold:
            step_score += 0.3
        
        # Determine the best classification
        scores = {
            'reach': reach_score,
            'pull': pull_score,
            'step': step_score
        }
        
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_type]
        
        # Generate description
        if best_type == 'reach':
            description = f"Reach movement (v={velocity:.3f}, arm_mv={arm_movement_score:.3f})"
        elif best_type == 'pull':
            description = f"Pull movement (v={velocity:.3f}, upward={upward_movement})"
        elif best_type == 'step':
            description = f"Step movement (v={velocity:.3f}, leg_mv={leg_movement_score:.3f})"
        else:
            description = f"Unknown movement (v={velocity:.3f})"
        
        return best_type, confidence, description
    
    def _detect_rhythm_events(self, timestamp: float, velocity: float, com_position: Tuple[float, float, float], features: MovementFeatures):
        """Detect rhythm events based on velocity patterns and movement classification."""
        
        # Skip if too close to previous event
        if self.rhythm_events and timestamp - self.rhythm_events[-1].timestamp < self.min_event_interval:
            return
        
        # Detect different types of events
        event_type = None
        confidence = 0.0
        description = ""
        
        # High velocity event - classify as reach, pull, or step
        if velocity > self.min_velocity_threshold:
            event_type, confidence, description = self._classify_movement_type(velocity, features)
        
        # Low velocity event (pause, hesitation)
        elif velocity < self.min_velocity_threshold:
            event_type = "pause"
            confidence = 1.0 - (velocity / self.min_velocity_threshold)
            description = f"Pause/hesitation (v={velocity:.3f})"
        
        if event_type:
            event = RhythmEvent(
                timestamp=timestamp,
                event_type=event_type,
                confidence=confidence,
                com_position=com_position,
                velocity=velocity,
                description=description,
                movement_features=features
            )
            self.rhythm_events.append(event)
    
    def _analyze_rhythm_patterns(self):
        """Analyze rhythm patterns in the recent movement history."""
        if len(self.velocity_history) < self.rhythm_window_size:
            return
        
        # Get recent velocity data
        recent_velocities = [v for _, v in self.velocity_history[-self.rhythm_window_size:]]
        recent_timestamps = [t for t, _ in self.velocity_history[-self.rhythm_window_size:]]
        
        # Calculate rhythm metrics
        tempo = self._calculate_tempo(recent_timestamps, recent_velocities)
        regularity = self._calculate_regularity(recent_velocities)
        rhythm_score = self._calculate_rhythm_score(recent_velocities, tempo, regularity)
        
        # Determine pattern type
        pattern_type = self._classify_pattern(recent_velocities, tempo, regularity)
        
        # Get events in this window
        window_start = recent_timestamps[0]
        window_end = recent_timestamps[-1]
        window_events = [e for e in self.rhythm_events 
                        if window_start <= e.timestamp <= window_end]
        
        # Create rhythm pattern
        pattern = RhythmPattern(
            pattern_type=pattern_type,
            start_time=window_start,
            end_time=window_end,
            events=window_events,
            rhythm_score=rhythm_score,
            tempo=tempo,
            regularity=regularity
        )
        
        self.rhythm_patterns.append(pattern)
    
    def _calculate_tempo(self, timestamps: List[float], velocities: List[float]) -> float:
        """Calculate the tempo (events per second) based on velocity peaks."""
        if len(velocities) < 3:
            return 0.0
        
        # Find velocity peaks (movement events)
        peaks, _ = find_peaks(velocities, height=self.min_velocity_threshold)
        
        if len(peaks) < 2:
            return 0.0
        
        # Calculate average time between peaks
        peak_times = [timestamps[i] for i in peaks]
        intervals = [peak_times[i+1] - peak_times[i] for i in range(len(peak_times)-1)]
        
        if intervals:
            avg_interval = np.mean(intervals)
            return float(1.0 / avg_interval if avg_interval > 0 else 0.0)
        
        return 0.0
    
    def _calculate_regularity(self, velocities: List[float]) -> float:
        """Calculate the regularity (consistency) of movement timing."""
        if len(velocities) < 3:
            return 0.0
        
        # Calculate velocity variance (lower variance = more regular)
        velocity_std = np.std(velocities)
        velocity_mean = np.mean(velocities)
        
        if velocity_mean == 0:
            return 0.0
        
        # Regularity is inverse of coefficient of variation
        cv = velocity_std / velocity_mean
        regularity = 1.0 / (1.0 + cv)  # Normalize to 0-1
        
        return float(max(0.0, min(1.0, regularity)))
    
    def _calculate_rhythm_score(self, velocities: List[float], tempo: float, regularity: float) -> float:
        """Calculate overall rhythm score (0-1)."""
        if len(velocities) < 3:
            return 0.0
        
        # Factors for rhythm score:
        # 1. Moderate tempo (not too fast, not too slow)
        # 2. High regularity
        # 3. Appropriate velocity range
        
        # Tempo score (prefer moderate tempo)
        optimal_tempo = 1.0  # events per second
        tempo_score = 1.0 - min(1.0, abs(tempo - optimal_tempo) / optimal_tempo)
        
        # Velocity range score
        velocity_mean = np.mean(velocities)
        velocity_score = 1.0 - min(1.0, abs(velocity_mean - 0.05) / 0.05)
        
        # Combine scores
        rhythm_score = (tempo_score * 0.3 + regularity * 0.4 + velocity_score * 0.3)
        
        return float(max(0.0, min(1.0, rhythm_score)))
    
    def _classify_pattern(self, velocities: List[float], tempo: float, regularity: float) -> str:
        """Classify the rhythm pattern type."""
        velocity_mean = np.mean(velocities)
        velocity_std = np.std(velocities)
        
        # High tempo, low regularity = burst
        if tempo > 1.5 and regularity < 0.5:
            return "burst"
        
        # Low tempo, high regularity = steady
        elif tempo < 0.5 and regularity > 0.7:
            return "steady"
        
        # Low velocity, low regularity = hesitant
        elif velocity_mean < 0.02 and regularity < 0.6:
            return "hesitant"
        
        # Moderate tempo, high regularity = flowing
        elif 0.5 <= tempo <= 1.5 and regularity > 0.6:
            return "flowing"
        
        # Default
        return "mixed"
    
    def get_current_rhythm_summary(self) -> Dict:
        """Get a summary of the current rhythm analysis."""
        if not self.rhythm_patterns:
            return {"status": "insufficient_data"}
        
        recent_patterns = self.rhythm_patterns[-5:]  # Last 5 patterns
        
        return {
            "current_pattern": recent_patterns[-1].pattern_type if recent_patterns else "unknown",
            "average_rhythm_score": np.mean([p.rhythm_score for p in recent_patterns]),
            "average_tempo": np.mean([p.tempo for p in recent_patterns]),
            "average_regularity": np.mean([p.regularity for p in recent_patterns]),
            "total_events": len(self.rhythm_events),
            "recent_events": len([e for e in self.rhythm_events if e.timestamp > recent_patterns[-1].start_time]) if recent_patterns else 0
        }
    
    def plot_rhythm_analysis(self, save_path: Optional[str] = None):
        """Plot rhythm analysis results."""
        if len(self.velocity_history) < 10:
            print("Insufficient data for rhythm analysis plot")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Extract data
        timestamps = [t for t, _ in self.velocity_history]
        velocities = [v for _, v in self.velocity_history]
        
        # Plot 1: Velocity over time
        axes[0].plot(timestamps, velocities, 'b-', alpha=0.7, label='COM Velocity')
        axes[0].axhline(y=self.min_velocity_threshold, color='r', linestyle='--', alpha=0.5, label='Min Threshold')
        axes[0].axhline(y=self.max_velocity_threshold, color='r', linestyle='--', alpha=0.5, label='Max Threshold')
        
        # Mark rhythm events
        for event in self.rhythm_events:
            color_map = {'reach': 'red', 'pull': 'orange', 'step': 'purple', 'pause': 'green', 'steady': 'blue'}
            color = color_map.get(event.event_type, 'gray')
            axes[0].scatter(event.timestamp, event.velocity, color=color, s=50, alpha=0.7)
        
        axes[0].set_ylabel('Velocity')
        axes[0].set_title('COM Velocity and Rhythm Events')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Rhythm patterns
        if self.rhythm_patterns:
            pattern_times = [(p.start_time, p.end_time) for p in self.rhythm_patterns]
            pattern_scores = [p.rhythm_score for p in self.rhythm_patterns]
            pattern_types = [p.pattern_type for p in self.rhythm_patterns]
            
            for i, ((start, end), score, pattern_type) in enumerate(zip(pattern_times, pattern_scores, pattern_types)):
                color_map = {'steady': 'green', 'burst': 'red', 'hesitant': 'orange', 'flowing': 'blue', 'mixed': 'gray'}
                color = color_map.get(pattern_type, 'gray')
                axes[1].barh(i, end - start, left=start, color=color, alpha=0.7, label=pattern_type if i == 0 else "")
                axes[1].text(start, i, f'{score:.2f}', va='center', ha='right', fontsize=8)
        
        axes[1].set_ylabel('Pattern')
        axes[1].set_title('Rhythm Patterns Over Time')
        axes[1].set_xlabel('Time (s)')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Rhythm metrics over time
        if self.rhythm_patterns:
            pattern_times = [(p.start_time + p.end_time) / 2 for p in self.rhythm_patterns]
            rhythm_scores = [p.rhythm_score for p in self.rhythm_patterns]
            tempos = [p.tempo for p in self.rhythm_patterns]
            regularities = [p.regularity for p in self.rhythm_patterns]
            
            axes[2].plot(pattern_times, rhythm_scores, 'b-', label='Rhythm Score', linewidth=2)
            axes[2].plot(pattern_times, tempos, 'r-', label='Tempo', alpha=0.7)
            axes[2].plot(pattern_times, regularities, 'g-', label='Regularity', alpha=0.7)
            axes[2].set_ylabel('Score')
            axes[2].set_title('Rhythm Metrics Over Time')
            axes[2].set_xlabel('Time (s)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Rhythm analysis plot saved to: {save_path}")
        
        plt.show()
    
    def get_movement_statistics(self) -> Dict:
        """Get detailed statistics about different movement types."""
        if not self.rhythm_events:
            return {"status": "no_events"}
        
        # Count events by type
        event_counts = {}
        event_velocities = {}
        event_confidences = {}
        
        for event in self.rhythm_events:
            event_type = event.event_type
            if event_type not in event_counts:
                event_counts[event_type] = 0
                event_velocities[event_type] = []
                event_confidences[event_type] = []
            
            event_counts[event_type] += 1
            event_velocities[event_type].append(event.velocity)
            event_confidences[event_type].append(event.confidence)
        
        # Calculate statistics
        stats = {}
        for event_type in event_counts:
            velocities = event_velocities[event_type]
            confidences = event_confidences[event_type]
            
            stats[event_type] = {
                "count": event_counts[event_type],
                "percentage": event_counts[event_type] / len(self.rhythm_events) * 100,
                "avg_velocity": np.mean(velocities),
                "avg_confidence": np.mean(confidences),
                "velocity_std": np.std(velocities),
                "confidence_std": np.std(confidences)
            }
        
        return stats
    
    def get_rhythm_recommendations(self) -> List[str]:
        """Get recommendations based on rhythm analysis."""
        summary = self.get_current_rhythm_summary()
        
        if summary.get("status") == "insufficient_data":
            return ["Need more movement data for rhythm analysis"]
        
        recommendations = []
        
        # Rhythm score recommendations
        rhythm_score = summary.get("average_rhythm_score", 0)
        if rhythm_score < 0.3:
            recommendations.append("Focus on developing consistent movement patterns")
        elif rhythm_score < 0.6:
            recommendations.append("Work on timing and flow between movements")
        else:
            recommendations.append("Excellent rhythm! Maintain this flow")
        
        # Tempo recommendations
        tempo = summary.get("average_tempo", 0)
        if tempo < 0.3:
            recommendations.append("Consider increasing movement tempo for better flow")
        elif tempo > 2.0:
            recommendations.append("Slow down slightly for better control and precision")
        
        # Regularity recommendations
        regularity = summary.get("average_regularity", 0)
        if regularity < 0.5:
            recommendations.append("Work on consistent timing between movements")
        
        # Pattern-specific recommendations
        current_pattern = summary.get("current_pattern", "")
        if current_pattern == "hesitant":
            recommendations.append("Reduce pauses and build confidence in movement")
        elif current_pattern == "burst":
            recommendations.append("Smooth out movements for better efficiency")
        elif current_pattern == "flowing":
            recommendations.append("Great flow! This is optimal climbing rhythm")
        
        # Movement type specific recommendations
        movement_stats = self.get_movement_statistics()
        if movement_stats.get("status") != "no_events":
            if "reach" in movement_stats:
                reach_confidence = movement_stats["reach"]["avg_confidence"]
                if reach_confidence < 0.5:
                    recommendations.append("Work on more confident reaching movements")
            
            if "pull" in movement_stats:
                pull_count = movement_stats["pull"]["count"]
                if pull_count < 3:
                    recommendations.append("Include more pulling movements in your sequence")
            
            if "step" in movement_stats:
                step_confidence = movement_stats["step"]["avg_confidence"]
                if step_confidence < 0.5:
                    recommendations.append("Focus on precise foot placement and stepping")
        
        return recommendations 