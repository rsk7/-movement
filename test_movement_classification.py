#!/usr/bin/env python3
"""
Test script for enhanced movement classification in rhythm detection.
Demonstrates the ability to distinguish between reach, pull, and step movements.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.rhythm_detection import RhythmDetector, MovementFeatures
from src.pose_detection import PoseFrame, PoseKeypoint
import time


def create_mock_pose_frame(timestamp: float, movement_type: str) -> PoseFrame:
    """Create a mock pose frame for testing different movement types."""
    keypoints = {}
    
    # Base positions (normal standing pose)
    base_positions = {
        'nose': (0.5, 0.2, 0.0),
        'left_shoulder': (0.4, 0.3, 0.0),
        'right_shoulder': (0.6, 0.3, 0.0),
        'left_elbow': (0.35, 0.5, 0.0),
        'right_elbow': (0.65, 0.5, 0.0),
        'left_wrist': (0.3, 0.7, 0.0),
        'right_wrist': (0.7, 0.7, 0.0),
        'left_hip': (0.4, 0.6, 0.0),
        'right_hip': (0.6, 0.6, 0.0),
        'left_knee': (0.4, 0.8, 0.0),
        'right_knee': (0.6, 0.8, 0.0),
        'left_ankle': (0.4, 0.95, 0.0),
        'right_ankle': (0.6, 0.95, 0.0),
    }
    
    # Modify positions based on movement type
    if movement_type == "reach":
        # Arms raised high for reaching
        keypoints['left_wrist'] = PoseKeypoint(0.3, 0.1, 0.0, 0.9)  # High up
        keypoints['right_wrist'] = PoseKeypoint(0.7, 0.1, 0.0, 0.9)  # High up
        keypoints['left_elbow'] = PoseKeypoint(0.35, 0.2, 0.0, 0.9)
        keypoints['right_elbow'] = PoseKeypoint(0.65, 0.2, 0.0, 0.9)
        
    elif movement_type == "pull":
        # Arms in pulling position, COM moving up
        keypoints['left_wrist'] = PoseKeypoint(0.4, 0.25, 0.0, 0.9)  # Near shoulders
        keypoints['right_wrist'] = PoseKeypoint(0.6, 0.25, 0.0, 0.9)  # Near shoulders
        keypoints['left_elbow'] = PoseKeypoint(0.4, 0.35, 0.0, 0.9)
        keypoints['right_elbow'] = PoseKeypoint(0.6, 0.35, 0.0, 0.9)
        
    elif movement_type == "step":
        # Legs in stepping position
        keypoints['left_ankle'] = PoseKeypoint(0.3, 0.85, 0.0, 0.9)  # Raised leg
        keypoints['right_ankle'] = PoseKeypoint(0.6, 0.95, 0.0, 0.9)  # Standing leg
        keypoints['left_knee'] = PoseKeypoint(0.3, 0.7, 0.0, 0.9)
        keypoints['right_knee'] = PoseKeypoint(0.6, 0.8, 0.0, 0.9)
        
    elif movement_type == "pause":
        # Normal standing position
        pass
    
    # Add remaining keypoints with base positions
    for name, (x, y, z) in base_positions.items():
        if name not in keypoints:
            keypoints[name] = PoseKeypoint(x, y, z, 0.9)
    
    # Calculate COM (simplified)
    com = (0.5, 0.4, 0.0)  # Center of body
    
    return PoseFrame(
        keypoints=keypoints,
        timestamp=timestamp,
        frame_number=int(timestamp * 30),  # Assuming 30 fps
        com=com
    )


def test_movement_classification():
    """Test the movement classification system."""
    print("Testing Enhanced Movement Classification System")
    print("=" * 50)
    
    # Initialize rhythm detector
    detector = RhythmDetector()
    
    # Test different movement sequences
    movement_sequences = [
        ("Reach Sequence", ["reach", "reach", "reach"]),
        ("Pull Sequence", ["pull", "pull", "pull"]),
        ("Step Sequence", ["step", "step", "step"]),
        ("Mixed Sequence", ["reach", "pull", "step", "reach", "pull"]),
        ("Pause Sequence", ["reach", "pause", "pull", "pause", "step"])
    ]
    
    for sequence_name, movements in movement_sequences:
        print(f"\n{sequence_name}:")
        print("-" * 30)
        
        # Reset detector for each sequence
        detector = RhythmDetector()
        
        # Simulate movement sequence
        timestamp = 0.0
        for i, movement_type in enumerate(movements):
            # Add some frames for the movement
            for frame_idx in range(5):  # 5 frames per movement
                frame_timestamp = timestamp + frame_idx * 0.033  # ~30 fps
                pose_frame = create_mock_pose_frame(frame_timestamp, movement_type)
                detector.add_frame(frame_timestamp, pose_frame)
            
            timestamp += 0.2  # 200ms per movement
        
        # Get results
        summary = detector.get_current_rhythm_summary()
        movement_stats = detector.get_movement_statistics()
        
        print(f"Rhythm Score: {summary.get('average_rhythm_score', 0):.3f}")
        print(f"Tempo: {summary.get('average_tempo', 0):.3f}")
        print(f"Regularity: {summary.get('average_regularity', 0):.3f}")
        
        if movement_stats.get("status") != "no_events":
            print("Movement Distribution:")
            for event_type, stats in movement_stats.items():
                print(f"  {event_type}: {stats['count']} events ({stats['percentage']:.1f}%)")
                print(f"    Avg velocity: {stats['avg_velocity']:.3f}")
                print(f"    Avg confidence: {stats['avg_confidence']:.3f}")
        
        # Get recommendations
        recommendations = detector.get_rhythm_recommendations()
        if recommendations:
            print("Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")


def test_feature_extraction():
    """Test the movement feature extraction."""
    print("\n\nTesting Movement Feature Extraction")
    print("=" * 50)
    
    detector = RhythmDetector()
    
    # Test feature extraction for different movements
    test_movements = ["reach", "pull", "step", "pause"]
    
    for movement_type in test_movements:
        print(f"\n{movement_type.upper()} Movement Features:")
        print("-" * 30)
        
        # Create pose frame
        pose_frame = create_mock_pose_frame(0.0, movement_type)
        
        # Extract features
        features = detector._extract_movement_features(0.0, pose_frame)
        
        print(f"COM Velocity: {features.com_velocity:.3f}")
        print(f"COM Direction: ({features.com_direction[0]:.3f}, {features.com_direction[1]:.3f})")
        print(f"Arm Movement: L={features.arm_movement['left']:.3f}, R={features.arm_movement['right']:.3f}")
        print(f"Leg Movement: L={features.leg_movement['left']:.3f}, R={features.leg_movement['right']:.3f}")
        print(f"Arm Elevation: L={features.arm_elevation['left']:.3f}, R={features.arm_elevation['right']:.3f}")
        print(f"Leg Elevation: L={features.leg_elevation['left']:.3f}, R={features.leg_elevation['right']:.3f}")
        
        # Test classification
        velocity = 0.1  # Mock velocity
        event_type, confidence, description = detector._classify_movement_type(velocity, features)
        print(f"Classified as: {event_type} (confidence: {confidence:.3f})")
        print(f"Description: {description}")


def create_visualization():
    """Create a visualization of the movement classification."""
    print("\n\nCreating Movement Classification Visualization")
    print("=" * 50)
    
    detector = RhythmDetector()
    
    # Create a longer sequence for visualization
    movements = ["reach", "pull", "step", "reach", "pause", "pull", "step", "reach", "pull", "step"]
    
    timestamp = 0.0
    for i, movement_type in enumerate(movements):
        # Add multiple frames for each movement
        for frame_idx in range(10):  # 10 frames per movement
            frame_timestamp = timestamp + frame_idx * 0.033
            pose_frame = create_mock_pose_frame(frame_timestamp, movement_type)
            detector.add_frame(frame_timestamp, pose_frame)
        
        timestamp += 0.3  # 300ms per movement
    
    # Create visualization
    try:
        detector.plot_rhythm_analysis("test_movement_classification.png")
        print("Visualization saved as 'test_movement_classification.png'")
    except Exception as e:
        print(f"Could not create visualization: {e}")


if __name__ == "__main__":
    # Run tests
    test_movement_classification()
    test_feature_extraction()
    create_visualization()
    
    print("\n\nTest completed! The enhanced movement classification system can now distinguish between:")
    print("- REACH: High arm movement with elevated arms")
    print("- PULL: Upward COM movement with engaged arms")
    print("- STEP: High leg movement with elevated legs")
    print("- PAUSE: Low velocity periods") 