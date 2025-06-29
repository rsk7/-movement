#!/usr/bin/env python3
"""
Video Conversion Script
Uses FFmpeg to convert .mov files to .mp4 format with various quality options.
"""

import subprocess
import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional


def convert_mov_to_mp4(input_path: str, output_path: str, 
                      quality: str = 'high', 
                      codec: str = 'h264',
                      overwrite: bool = False) -> bool:
    """
    Convert .mov file to .mp4 using FFmpeg.
    
    Args:
        input_path: Path to input .mov file
        output_path: Path for output .mp4 file
        quality: Quality preset ('high', 'medium', 'low', 'fast')
        codec: Video codec to use ('h264', 'h265', 'vp9')
        overwrite: Whether to overwrite existing output file
        
    Returns:
        True if conversion successful, False otherwise
    """
    
    # Quality presets
    quality_presets = {
        'high': {
            'h264': ['-c:v', 'libx264', '-preset', 'slow', '-crf', '18'],
            'h265': ['-c:v', 'libx265', '-preset', 'slow', '-crf', '20'],
            'vp9': ['-c:v', 'libvpx-vp9', '-crf', '20', '-b:v', '0']
        },
        'medium': {
            'h264': ['-c:v', 'libx264', '-preset', 'medium', '-crf', '23'],
            'h265': ['-c:v', 'libx265', '-preset', 'medium', '-crf', '25'],
            'vp9': ['-c:v', 'libvpx-vp9', '-crf', '25', '-b:v', '0']
        },
        'low': {
            'h264': ['-c:v', 'libx264', '-preset', 'fast', '-crf', '28'],
            'h265': ['-c:v', 'libx265', '-preset', 'fast', '-crf', '30'],
            'vp9': ['-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0']
        },
        'fast': {
            'h264': ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '30'],
            'h265': ['-c:v', 'libx265', '-preset', 'ultrafast', '-crf', '32'],
            'vp9': ['-c:v', 'libvpx-vp9', '-crf', '35', '-b:v', '0', '-deadline', 'realtime']
        }
    }
    
    # Validate inputs
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return False
    
    if quality not in quality_presets:
        print(f"❌ Invalid quality preset: {quality}")
        return False
    
    if codec not in quality_presets[quality]:
        print(f"❌ Codec {codec} not supported for quality {quality}")
        return False
    
    # Check if output file exists
    if os.path.exists(output_path) and not overwrite:
        print(f"❌ Output file already exists: {output_path}")
        print("Use --overwrite to overwrite existing files")
        return False
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build FFmpeg command
    cmd = ['ffmpeg', '-i', input_path]
    
    # Add video codec options
    cmd.extend(quality_presets[quality][codec])
    
    # Add audio codec (copy if possible, otherwise AAC)
    cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
    
    # Add output options
    cmd.extend(['-movflags', '+faststart'])  # Optimize for web streaming
    
    # Add overwrite flag if needed
    if overwrite:
        cmd.append('-y')
    else:
        cmd.append('-n')  # Don't overwrite
    
    # Add output file
    cmd.append(output_path)
    
    print(f"🔄 Converting {input_path} to {output_path}")
    print(f"   Quality: {quality}, Codec: {codec}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Conversion completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)


def batch_convert(input_dir: str, output_dir: str, 
                 quality: str = 'high', 
                 codec: str = 'h264',
                 overwrite: bool = False) -> bool:
    """
    Convert all .mov files in a directory to .mp4.
    
    Args:
        input_dir: Directory containing .mov files
        output_dir: Directory for output .mp4 files
        quality: Quality preset
        codec: Video codec
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if all conversions successful, False otherwise
    """
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all .mov files
    mov_files = list(Path(input_dir).glob("*.mov"))
    mov_files.extend(Path(input_dir).glob("*.MOV"))
    
    if not mov_files:
        print(f"❌ No .mov files found in {input_dir}")
        return False
    
    print(f"📁 Found {len(mov_files)} .mov files to convert")
    
    success_count = 0
    total_original_size = 0
    total_converted_size = 0
    
    for mov_file in mov_files:
        # Create output filename
        output_file = Path(output_dir) / f"{mov_file.stem}.mp4"
        
        print(f"\n🔄 Converting: {mov_file.name}")
        
        # Get original file size
        original_size = get_file_size_mb(str(mov_file))
        total_original_size += original_size
        
        # Convert file
        if convert_mov_to_mp4(str(mov_file), str(output_file), quality, codec, overwrite):
            success_count += 1
            
            # Get converted file size
            if os.path.exists(str(output_file)):
                converted_size = get_file_size_mb(str(output_file))
                total_converted_size += converted_size
                compression_ratio = (1 - converted_size / original_size) * 100
                print(f"   Size: {original_size:.1f}MB → {converted_size:.1f}MB ({compression_ratio:.1f}% smaller)")
        else:
            print(f"   ❌ Failed to convert {mov_file.name}")
    
    # Summary
    print(f"\n📊 Conversion Summary:")
    print(f"   Successfully converted: {success_count}/{len(mov_files)} files")
    print(f"   Total original size: {total_original_size:.1f}MB")
    print(f"   Total converted size: {total_converted_size:.1f}MB")
    if total_original_size > 0:
        total_compression = (1 - total_converted_size / total_original_size) * 100
        print(f"   Overall compression: {total_compression:.1f}%")
    
    return success_count == len(mov_files)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert .mov files to .mp4 using FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_video.py input.mov output.mp4
  python convert_video.py input.mov output.mp4 --quality medium
  python convert_video.py input.mov output.mp4 --codec h265 --quality high
  python convert_video.py --batch input_folder output_folder --quality fast
  python convert_video.py --batch input_folder output_folder --overwrite
        """
    )
    
    # Input/output arguments
    parser.add_argument('input', nargs='?', help='Input .mov file or directory (for batch mode)')
    parser.add_argument('output', nargs='?', help='Output .mp4 file or directory (for batch mode)')
    
    # Batch mode
    parser.add_argument('--batch', action='store_true',
                       help='Batch convert all .mov files in input directory')
    
    # Quality and codec options
    parser.add_argument('--quality', choices=['high', 'medium', 'low', 'fast'], default='high',
                       help='Quality preset (default: high)')
    parser.add_argument('--codec', choices=['h264', 'h265', 'vp9'], default='h264',
                       help='Video codec (default: h264)')
    
    # Other options
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch:
        if not args.input or not args.output:
            print("❌ Batch mode requires both input and output directories")
            sys.exit(1)
        
        success = batch_convert(args.input, args.output, args.quality, args.codec, args.overwrite)
        
    else:
        if not args.input or not args.output:
            print("❌ Single file mode requires both input and output files")
            sys.exit(1)
        
        success = convert_mov_to_mp4(args.input, args.output, args.quality, args.codec, args.overwrite)
    
    if success:
        print("\n🎉 All operations completed successfully!")
    else:
        print("\n❌ Some operations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 