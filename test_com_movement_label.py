#!/usr/bin/env python3
"""
Test script to demonstrate COM movement type labeling.
Shows how movement types (REACH, PULL, STEP, PAUSE) are displayed next to the COM position.
"""

import numpy as np
import cv2
from src.visualization import PoseVisualizer
from src.pose_detection import PoseFrame, PoseKeypoint
from src.rhythm_detection import RhythmEvent


def create_test_frame_with_movement(movement_type: str) -> np.ndarray:
    """Create a test frame with a climber and movement type."""
    # Create a simple test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple climber silhouette
    # Head
    cv2.circle(frame, (320, 100), 20, (255, 255, 255), -1)
    
    # Body
    cv2.rectangle(frame, (310, 120), (330, 200), (255, 255, 255), -1)
    
    # Arms (positioned based on movement type)
    if movement_type == "reach":
        # Arms raised high
        cv2.line(frame, (310, 130), (280, 80), (255, 255, 255), 3)
        cv2.line(frame, (330, 130), (360, 80), (255, 255, 255), 3)
    elif movement_type == "pull":
        # Arms in pulling position
        cv2.line(frame, (310, 130), (290, 110), (255, 255, 255), 3)
        cv2.line(frame, (330, 130), (350, 110), (255, 255, 255), 3)
    elif movement_type == "step":
        # One leg raised
        cv2.line(frame, (310, 200), (300, 250), (255, 255, 255), 3)
        cv2.line(frame, (330, 200), (340, 250), (255, 255, 255), 3)
    else:
        # Normal position
        cv2.line(frame, (310, 130), (290, 150), (255, 255, 255), 3)
        cv2.line(frame, (330, 130), (350, 150), (255, 255, 255), 3)
    
    # Legs
    cv2.line(frame, (310, 200), (300, 280), (255, 255, 255), 3)
    cv2.line(frame, (330, 200), (340, 280), (255, 255, 255), 3)
    
    return frame


def create_mock_pose_frame(movement_type: str) -> PoseFrame:
    """Create a mock pose frame with COM and rhythm events."""
    # Create basic keypoints
    keypoints = {
        'nose': PoseKeypoint(0.5, 0.2, 0.0, 0.9),
        'left_shoulder': PoseKeypoint(0.4, 0.3, 0.0, 0.9),
        'right_shoulder': PoseKeypoint(0.6, 0.3, 0.0, 0.9),
        'left_elbow': PoseKeypoint(0.35, 0.5, 0.0, 0.9),
        'right_elbow': PoseKeypoint(0.65, 0.5, 0.0, 0.9),
        'left_wrist': PoseKeypoint(0.3, 0.7, 0.0, 0.9),
        'right_wrist': PoseKeypoint(0.7, 0.7, 0.0, 0.9),
        'left_hip': PoseKeypoint(0.4, 0.6, 0.0, 0.9),
        'right_hip': PoseKeypoint(0.6, 0.6, 0.0, 0.9),
        'left_knee': PoseKeypoint(0.4, 0.8, 0.0, 0.9),
        'right_knee': PoseKeypoint(0.6, 0.8, 0.0, 0.9),
        'left_ankle': PoseKeypoint(0.4, 0.95, 0.0, 0.9),
        'right_ankle': PoseKeypoint(0.6, 0.95, 0.0, 0.9),
    }
    
    # Adjust keypoints based on movement type
    if movement_type == "reach":
        keypoints['left_wrist'] = PoseKeypoint(0.3, 0.1, 0.0, 0.9)
        keypoints['right_wrist'] = PoseKeypoint(0.7, 0.1, 0.0, 0.9)
    elif movement_type == "pull":
        keypoints['left_wrist'] = PoseKeypoint(0.4, 0.25, 0.0, 0.9)
        keypoints['right_wrist'] = PoseKeypoint(0.6, 0.25, 0.0, 0.9)
    elif movement_type == "step":
        keypoints['left_ankle'] = PoseKeypoint(0.3, 0.85, 0.0, 0.9)
    
    # Create pose frame
    pose_frame = PoseFrame(
        keypoints=keypoints,
        timestamp=0.0,
        frame_number=0,
        com=(0.5, 0.4, 0.0)  # Center of body
    )
    
    # Add rhythm event
    rhythm_event = RhythmEvent(
        timestamp=0.0,
        event_type=movement_type,
        confidence=0.8,
        com_position=(0.5, 0.4, 0.0),
        velocity=0.1,
        description=f"{movement_type.capitalize()} movement"
    )
    
    # Add rhythm events as an attribute (using setattr to avoid linter error)
    setattr(pose_frame, 'rhythm_events', [rhythm_event])
    
    return pose_frame


def test_com_movement_labeling():
    """Test the COM movement type labeling feature."""
    print("Testing COM Movement Type Labeling")
    print("=" * 40)
    
    # Initialize visualizer
    visualizer = PoseVisualizer()
    
    # Test different movement types
    movement_types = ["reach", "pull", "step", "pause"]
    
    for movement_type in movement_types:
        print(f"\nTesting {movement_type.upper()} movement:")
        
        # Create test frame
        frame = create_test_frame_with_movement(movement_type)
        
        # Create pose frame with movement type
        pose_frame = create_mock_pose_frame(movement_type)
        
        # Create overlay with COM and movement type
        overlay_frame = visualizer.create_overlay_frame(
            frame=frame,
            pose_frame=pose_frame,
            show_com=True,
            show_rhythm=True,
            show_trails=False,
            show_angles=False
        )
        
        # Save the result
        output_filename = f"test_com_{movement_type}.png"
        cv2.imwrite(output_filename, overlay_frame)
        print(f"  Saved: {output_filename}")
        
        # Also test the COM drawing function directly
        com_frame = frame.copy()
        if pose_frame.com is not None:
            com_frame = visualizer.draw_center_of_mass(
                com_frame, 
                pose_frame.com, 
                640, 480, 
                movement_type=movement_type
            )
        
        direct_output_filename = f"test_com_direct_{movement_type}.png"
        cv2.imwrite(direct_output_filename, com_frame)
        print(f"  Direct COM test: {direct_output_filename}")


def create_comparison_image():
    """Create a comparison image showing all movement types."""
    print("\nCreating comparison image...")
    
    visualizer = PoseVisualizer()
    
    # Create a large frame to hold all movement types
    comparison_frame = np.zeros((480, 1280, 3), dtype=np.uint8)
    
    # Create sections for each movement type
    movement_types = ["reach", "pull", "step", "pause"]
    section_width = 1280 // 4
    
    for i, movement_type in enumerate(movement_types):
        # Create section frame
        section_frame = create_test_frame_with_movement(movement_type)
        section_frame = cv2.resize(section_frame, (section_width, 480))
        
        # Create pose frame
        pose_frame = create_mock_pose_frame(movement_type)
        
        # Add overlay
        overlay_section = visualizer.create_overlay_frame(
            frame=section_frame,
            pose_frame=pose_frame,
            show_com=True,
            show_rhythm=True,
            show_trails=False,
            show_angles=False
        )
        
        # Add movement type title
        cv2.putText(overlay_section, movement_type.upper(), 
                   (section_width//2 - 50, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Copy to comparison frame
        x_start = i * section_width
        comparison_frame[:, x_start:x_start + section_width] = overlay_section
    
    # Save comparison
    cv2.imwrite("test_com_comparison.png", comparison_frame)
    print("  Saved: test_com_comparison.png")


if __name__ == "__main__":
    test_com_movement_labeling()
    create_comparison_image()
    
    print("\n" + "=" * 40)
    print("Test completed!")
    print("\nThe COM movement type labeling feature now shows:")
    print("- REACH: Red label below COM")
    print("- PULL: Orange label below COM") 
    print("- STEP: Purple label below COM")
    print("- PAUSE: Green label below COM")
    print("\nGenerated test images show the movement type labels positioned below the COM marker.") 