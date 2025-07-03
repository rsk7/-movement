#!/usr/bin/env python3
"""
Batch processing script for climbing videos.
Processes all videos in input_videos folder with pose tracking.
"""

import subprocess
import os
from pathlib import Path
import argparse


def process_video(input_video: str, output_video: str, quality_factor: float = 0.1) -> bool:
    """
    Process a single video with pose tracking.
    
    Args:
        input_video: Path to input video
        output_video: Path for output video
        quality_factor: Quality factor for processing
        
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'python', '-m', 'src.main',
        '--input', input_video,
        '--output', output_video,
        '--quality-factor', str(quality_factor),
        '--show-com',
        '--show-trails',
        '--show-motion-blur',
        '--show-energy',
        '--show-angles',
        '--show-rhythm'
    ]
    
    print(f"🔄 Processing: {os.path.basename(input_video)}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Successfully processed: {os.path.basename(input_video)}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to process {os.path.basename(input_video)}: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False


def batch_process_videos(input_dir: str = "data/input_videos", 
                        output_dir: str = "data/output_videos",
                        quality_factor: float = 0.1,
                        overwrite: bool = False) -> bool:
    """
    Process all videos in input directory.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory for output videos
        quality_factor: Quality factor for processing
        overwrite: Whether to overwrite existing output files
        
    Returns:
        True if all videos processed successfully, False otherwise
    """
    
    # Ensure directories exist
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")
    
    # Find all video files
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"❌ No video files found in {input_dir}")
        return False
    
    print(f"📁 Found {len(video_files)} video files to process")
    
    success_count = 0
    total_count = len(video_files)
    
    for video_file in video_files:
        # Skip .gitkeep file
        if video_file.name == '.gitkeep':
            continue
            
        # Create output filename
        output_filename = f"{video_file.stem}_joint_angles.mp4"
        output_file = output_path / output_filename
        
        # Check if output already exists
        if output_file.exists() and not overwrite:
            print(f"⏭️  Skipping {video_file.name} (output already exists)")
            continue
        
        # Process the video
        if process_video(str(video_file), str(output_file), quality_factor):
            success_count += 1
        
        print()  # Add spacing between videos
    
    # Summary
    print(f"📊 Processing Summary:")
    print(f"   Successfully processed: {success_count}/{total_count} videos")
    print(f"   Failed: {total_count - success_count} videos")
    
    if success_count == total_count:
        print("🎉 All videos processed successfully!")
        return True
    else:
        print("⚠️  Some videos failed to process")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch process climbing videos with pose tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process_videos.py
  python batch_process_videos.py --quality-factor 0.5
  python batch_process_videos.py --overwrite
  python batch_process_videos.py --input my_videos --output my_output
        """
    )
    
    parser.add_argument('--input', default='data/input_videos',
                       help='Input directory containing videos (default: data/input_videos)')
    parser.add_argument('--output', default='data/output_videos',
                       help='Output directory for processed videos (default: data/output_videos)')
    parser.add_argument('--quality-factor', type=float, default=0.1,
                       help='Quality factor for processing (default: 0.1)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files')
    
    args = parser.parse_args()
    
    success = batch_process_videos(
        input_dir=args.input,
        output_dir=args.output,
        quality_factor=args.quality_factor,
        overwrite=args.overwrite
    )
    
    if not success:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 