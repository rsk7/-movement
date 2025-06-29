#!/usr/bin/env python3
"""
Simple camera test script
"""

import cv2
import sys

def test_camera(camera_id):
    print(f"Testing camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_id}")
        return False
    
    print(f"✅ Camera {camera_id} opened successfully")
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    
    # Try to read a frame
    ret, frame = cap.read()
    if ret:
        print(f"✅ Successfully read frame from camera {camera_id}")
        print(f"   Frame shape: {frame.shape}")
    else:
        print(f"❌ Failed to read frame from camera {camera_id}")
    
    cap.release()
    return ret

if __name__ == "__main__":
    print("Testing available cameras...")
    
    # Test cameras 0-3
    for i in range(4):
        test_camera(i)
        print()
    
    print("Camera test complete!") 