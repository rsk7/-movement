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
from src.pose_detection import PoseFrame


@dataclass
class RhythmEvent:
    """Represents a rhythm event in climbing movement."""
    timestamp: float
    event_type: str  # 'reach', 'pull', 'step', 'release', 'pause'
    confidence: float
    com_position: Tuple[float, float, float]
    velocity: float
    description: str


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
        self.rhythm_events: List[RhythmEvent] = []
        self.rhythm_patterns: List[RhythmPattern] = []
    
    def add_frame(self, timestamp: float, pose_frame: PoseFrame):
        """Add a new frame for rhythm analysis."""
        if pose_frame.com is None:
            return
            
        # Store COM position and timestamp
        self.com_history.append((timestamp, pose_frame.com))
        
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
                
                # Detect rhythm events based on velocity changes
                self._detect_rhythm_events(timestamp, velocity, pose_frame.com)
        
        # Analyze rhythm patterns periodically
        if len(self.velocity_history) >= self.rhythm_window_size:
            self._analyze_rhythm_patterns()
    
    def _detect_rhythm_events(self, timestamp: float, velocity: float, com_position: Tuple[float, float, float]):
        """Detect rhythm events based on velocity patterns."""
        
        # Skip if too close to previous event
        if self.rhythm_events and timestamp - self.rhythm_events[-1].timestamp < self.min_event_interval:
            return
        
        # Detect different types of events
        event_type = None
        confidence = 0.0
        description = ""
        
        # High velocity event (reach, pull, step)
        if velocity > self.max_velocity_threshold:
            event_type = "reach"
            confidence = min(1.0, velocity / (self.max_velocity_threshold * 2))
            description = f"Fast movement (v={velocity:.3f})"
        
        # Low velocity event (pause, hesitation)
        elif velocity < self.min_velocity_threshold:
            event_type = "pause"
            confidence = 1.0 - (velocity / self.min_velocity_threshold)
            description = f"Pause/hesitation (v={velocity:.3f})"
        
        # Medium velocity event (steady movement)
        elif self.min_velocity_threshold <= velocity <= self.max_velocity_threshold:
            event_type = "steady"
            confidence = 0.7
            description = f"Steady movement (v={velocity:.3f})"
        
        if event_type:
            event = RhythmEvent(
                timestamp=timestamp,
                event_type=event_type,
                confidence=confidence,
                com_position=com_position,
                velocity=velocity,
                description=description
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
            color_map = {'reach': 'red', 'pause': 'green', 'steady': 'blue'}
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
        
        return recommendations 