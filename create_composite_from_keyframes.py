#!/usr/bin/env python3
"""
Script to create a composite image from keyframe images.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse

def create_composite_from_keyframes(keyframes_dir: str, output_path: str, alpha: float = 0.4):
    """
    Create a composite image from keyframe images with transparency overlay.
    
    Args:
        keyframes_dir: Directory containing keyframe images
        output_path: Path for output composite image
        alpha: Transparency factor (0.0-1.0, lower = more transparent)
    """
    keyframes_path = Path(keyframes_dir)
    
    if not keyframes_path.exists():
        print(f"Error: Keyframes directory not found: {keyframes_dir}")
        return False
    
    # Get all PNG images in the directory
    image_files = sorted([f for f in keyframes_path.glob("*.png")])
    
    if not image_files:
        print(f"Error: No PNG images found in {keyframes_dir}")
        return False
    
    print(f"Found {len(image_files)} keyframe images")
    
    # Load the first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"Error: Could not load first image: {image_files[0]}")
        return False
    
    height, width = first_image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Create composite (start with first image)
    composite = first_image.astype(np.float32)
    
    # Overlay each subsequent image with transparency
    for i, image_file in enumerate(image_files[1:], 1):
        print(f"Processing {image_file.name} ({i+1}/{len(image_files)})")
        
        # Load image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Warning: Could not load {image_file.name}, skipping...")
            continue
        
        # Resize if dimensions don't match
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        
        # Convert to float for blending
        image = image.astype(np.float32)
        
        # Blend with composite
        composite = cv2.addWeighted(composite, 1 - alpha, image, alpha, 0)
    
    # Convert back to uint8
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    
    # Add title and information
    title_text = f"Climbing Sequence Composite - {len(image_files)} Keyframes"
    cv2.putText(composite, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save composite image
    cv2.imwrite(output_path, composite)
    print(f"Composite image saved to: {output_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Create composite image from keyframe images")
    parser.add_argument("keyframes_dir", help="Directory containing keyframe images")
    parser.add_argument("output_path", help="Path for output composite image")
    parser.add_argument("--alpha", type=float, default=0.4, 
                       help="Transparency factor (0.0-1.0, default: 0.4)")
    
    args = parser.parse_args()
    
    success = create_composite_from_keyframes(args.keyframes_dir, args.output_path, args.alpha)
    
    if success:
        print("✅ Composite creation completed!")
    else:
        print("❌ Composite creation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 