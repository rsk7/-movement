#!/usr/bin/env python3
"""
Video Information Script
Uses FFmpeg to extract and display video metadata in a user-friendly format.
"""

import subprocess
import json
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any


def run_ffprobe(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Run ffprobe to get video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary containing video information or None if failed
    """
    try:
        # Run ffprobe to get JSON output
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing ffprobe output: {e}")
        return None


def parse_video_info(ffprobe_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Parse ffprobe output into useful video information.
    
    Args:
        ffprobe_data: Raw ffprobe JSON output
        
    Returns:
        Parsed video information or None if failed
    """
    if not ffprobe_data:
        return None
    
    info: Dict[str, Any] = {
        'file_path': None,
        'format': {},
        'video_stream': {},
        'audio_stream': {},
        'duration': None,
        'file_size': None,
        'bitrate': None
    }
    
    # Format information
    if 'format' in ffprobe_data:
        format_info = ffprobe_data['format']
        info['format'] = {
            'format_name': format_info.get('format_name', 'Unknown'),
            'duration': float(format_info.get('duration', 0)),
            'size': int(format_info.get('size', 0)),
            'bit_rate': int(float(format_info.get('bit_rate', 0)))
        }
        info['duration'] = info['format']['duration']
        info['file_size'] = info['format']['size']
        info['bitrate'] = info['format']['bit_rate']
    
    # Stream information
    if 'streams' in ffprobe_data:
        for stream in ffprobe_data['streams']:
            if stream.get('codec_type') == 'video':
                info['video_stream'] = {
                    'codec': stream.get('codec_name', 'Unknown'),
                    'width': int(stream.get('width', 0)),
                    'height': int(stream.get('height', 0)),
                    'fps': parse_fps(stream.get('r_frame_rate', '0/1')),
                    'bitrate': int(float(stream.get('bit_rate', 0))),
                    'pixel_format': stream.get('pix_fmt', 'Unknown'),
                    'color_space': stream.get('color_space', 'Unknown')
                }
            elif stream.get('codec_type') == 'audio':
                info['audio_stream'] = {
                    'codec': stream.get('codec_name', 'Unknown'),
                    'sample_rate': int(stream.get('sample_rate', 0)),
                    'channels': int(stream.get('channels', 0)),
                    'bitrate': int(float(stream.get('bit_rate', 0)))
                }
    
    return info


def parse_fps(fps_string: str) -> float:
    """
    Parse frame rate string (e.g., "30/1") to float.
    
    Args:
        fps_string: Frame rate string from ffprobe
        
    Returns:
        Frame rate as float
    """
    try:
        if '/' in fps_string:
            num, den = map(int, fps_string.split('/'))
            return num / den if den != 0 else 0
        else:
            return float(fps_string)
    except (ValueError, ZeroDivisionError):
        return 0


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def format_file_size(bytes_size: float) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        bytes_size: File size in bytes
        
    Returns:
        Formatted file size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def print_video_info(info: Optional[Dict[str, Any]], detailed: bool = False):
    """
    Print video information in a formatted way.
    
    Args:
        info: Video information dictionary
        detailed: Whether to show detailed information
    """
    if not info:
        print("❌ Could not retrieve video information")
        return
    
    print("📹 Video Information")
    print("=" * 50)
    
    # Basic information
    if info['video_stream']:
        video = info['video_stream']
        print(f"Resolution:     {video['width']}x{video['height']}")
        print(f"Frame Rate:     {video['fps']:.2f} fps")
        print(f"Video Codec:    {video['codec']}")
    
    if info['duration']:
        print(f"Duration:       {format_duration(info['duration'])}")
    
    if info['file_size']:
        print(f"File Size:      {format_file_size(info['file_size'])}")
    
    if info['bitrate']:
        print(f"Bitrate:        {info['bitrate'] // 1000} kbps")
    
    if info['audio_stream']:
        audio = info['audio_stream']
        print(f"Audio Codec:    {audio['codec']}")
        print(f"Sample Rate:    {audio['sample_rate']} Hz")
        print(f"Channels:       {audio['channels']}")
    
    # Detailed information
    if detailed and info['video_stream']:
        print("\n🔍 Detailed Information")
        print("-" * 30)
        video = info['video_stream']
        print(f"Pixel Format:   {video['pixel_format']}")
        print(f"Color Space:    {video['color_space']}")
        if video['bitrate']:
            print(f"Video Bitrate:  {video['bitrate'] // 1000} kbps")
    
    if detailed and info['format']:
        print(f"Container:      {info['format']['format_name']}")
    
    # Processing recommendations
    print("\n💡 Processing Recommendations")
    print("-" * 30)
    
    if info['video_stream']:
        width = info['video_stream']['width']
        fps = info['video_stream']['fps']
        
        if width > 1920:
            print(f"• High resolution ({width}p) detected")
            print(f"  Consider: --quality-factor 0.5")
        
        if fps > 30:
            print(f"• High frame rate ({fps:.1f} fps) detected")
            print(f"  Consider: --target-fps 30")
        
        if info['file_size'] and info['file_size'] > 500 * 1024 * 1024:  # 500MB
            print(f"• Large file ({format_file_size(info['file_size'])}) detected")
            print(f"  Consider: --quality-factor 0.3 --target-fps 15")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Get video information using FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_info.py video.mp4
  python video_info.py video.mp4 --detailed
  python video_info.py *.mp4
        """
    )
    
    parser.add_argument('video_path', help='Path to video file(s)')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Show detailed information')
    parser.add_argument('--json', action='store_true',
                       help='Output raw JSON from ffprobe')
    
    args = parser.parse_args()
    
    # Handle multiple files
    video_files = list(Path('.').glob(args.video_path)) if '*' in args.video_path else [Path(args.video_path)]
    
    for video_file in video_files:
        if not video_file.exists():
            print(f"❌ File not found: {video_file}")
            continue
        
        print(f"\n📁 {video_file}")
        print("=" * 60)
        
        # Get video information
        ffprobe_data = run_ffprobe(str(video_file))
        
        if args.json:
            print(json.dumps(ffprobe_data, indent=2))
        else:
            info = parse_video_info(ffprobe_data)
            if info:
                info['file_path'] = str(video_file)
                print_video_info(info, args.detailed)
            else:
                print("❌ Could not parse video information")


if __name__ == "__main__":
    main() 