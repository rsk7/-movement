# Climbing Motion Tracker

A computer vision application that analyzes climbing videos and superimposes motion tracking visualizations to help climbers analyze their technique and movement patterns.

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have FFmpeg installed on your system

## Usage

### Basic Usage

```bash
python src/main.py --input path/to/your/climbing_video.mp4 --output path/to/output_video.mp4
```

### Advanced Options

```bash
python -m src.main \
    --input input_video.mp4 \
    --output output_video.mp4 \
    --show-trails \
    --show-angles \
    --trail-length 30 \
    --quality-factor 0.1 \
    --show-energy
```

### Video Information Tool

Get detailed information about your video files before processing:

```bash
# Basic video info
python video_info.py your_video.mp4

# Detailed information
python video_info.py your_video.mp4 --detailed

# Raw JSON output
python video_info.py your_video.mp4 --json

# Multiple files
python video_info.py *.mp4
```

The video info tool will also provide processing recommendations based on your video's characteristics.

### Video Conversion Tool

Convert .mov files to .mp4 format for better compatibility:

```bash
# Single file conversion
python convert_video.py input.mov output.mp4

# High quality conversion
python convert_video.py input.mov output.mp4 --quality high --codec h265

# Fast conversion for quick processing
python convert_video.py input.mov output.mp4 --quality fast

# Batch convert all .mov files in a folder
python convert_video.py --batch input_folder output_folder

# Overwrite existing files
python convert_video.py --batch input_folder output_folder --overwrite
```

**Quality Presets:**

- `high`: Best quality, slower encoding (recommended for final output)
- `medium`: Good balance of quality and speed
- `low`: Smaller files, faster encoding
- `fast`: Fastest encoding, lower quality (good for testing)

**Supported Codecs:**

- `h264`: Widely compatible (default)
- `h265`: Better compression, newer devices
- `vp9`: Open source, good for web

### Parameters

- `--input`: Path to input climbing video
- `--output`: Path for output video with tracking
- `--show-trails`: Enable motion trails (shows movement history)
- `--show-angles`: Display joint angles
- `--trail-length`: Number of frames to show in motion trails (default: 30)
- `--quality-factor`: Scale factor for processing resolution (0.1-1.0, default: 1.0)
- `--target-width`: Target width for processing (maintains aspect ratio)
- `--target-height`: Target height for processing (maintains aspect ratio)
- `--target-fps`: Target frame rate for processing (default: original fps)
- `--model-complexity`: MediaPipe model complexity (0, 1, or 2, default: 1)
- `--min-detection-confidence`: Minimum confidence for pose detection (default: 0.5)
- `--min-tracking-confidence`: Minimum confidence for pose tracking (default: 0.5)
- `--no-smoothing`: Disable pose smoothing

## Project Structure

```
climbing-tracker/
├── src/
│   ├── pose_detection.py    # MediaPipe pose estimation
│   ├── video_processor.py   # Video I/O and frame processing
│   ├── visualization.py     # Drawing overlays and visualizations
│   └── main.py             # Main application entry point
├── data/
│   ├── input_videos/       # Place your climbing videos here
│   └── output_videos/      # Processed videos will be saved here
├── requirements.txt
└── README.md
```

## How It Works

1. **Video Input**: The system reads your climbing video frame by frame
2. **Pose Detection**: MediaPipe analyzes each frame to detect body keypoints
3. **Data Processing**: Pose data is smoothed and processed for consistency
4. **Visualization**: Custom overlays are drawn on each frame
5. **Video Output**: Processed frames are combined into a new video with tracking

## Supported Video Formats

- MP4, MOV, AVI, MKV, and other common formats
- Recommended resolution: 720p or higher for better pose detection
- Frame rate: Any standard frame rate (24fps, 30fps, 60fps)

## Tips for Best Results

- Ensure good lighting in your climbing videos
- Wear contrasting clothing to improve pose detection
- Keep the climber in frame throughout the video
- Avoid rapid camera movements during recording

## Handling High-Quality Videos

For high-resolution (4K, 1080p) or high-frame-rate (60fps+) videos, consider using preprocessing options to improve performance:

### Performance Recommendations

**For 4K videos:**

```bash
python src/main.py --input 4k_video.mp4 --output tracked.mp4 --quality-factor 0.5 --target-fps 30
```

**For 60fps videos:**

```bash
python src/main.py --input 60fps_video.mp4 --output tracked.mp4 --target-fps 30
```

**For very large files:**

```bash
python src/main.py --input large_video.mp4 --output tracked.mp4 --quality-factor 0.3 --target-fps 15
```

### Processing Speed vs Quality Trade-offs

- **Quality Factor 1.0**: Full resolution, slowest processing
- **Quality Factor 0.5**: Half resolution, ~4x faster processing
- **Quality Factor 0.3**: 30% resolution, ~10x faster processing
- **Target FPS 30**: Good balance for most climbing analysis
- **Target FPS 15**: Faster processing, still captures key movements
