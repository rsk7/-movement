# Climbing Motion Tracker

A computer vision application that analyzes climbing videos and superimposes motion tracking visualizations to help climbers analyze their technique and movement patterns.

## Features

- **Pose Detection**: Tracks key body points (shoulders, elbows, wrists, hips, knees, ankles)
- **Motion Visualization**: Superimposes skeleton overlays and motion trails on climbing videos
- **Performance Analysis**: Calculates movement metrics and joint angles
- **Video Processing**: Maintains original video quality while adding tracking data

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
python src/main.py \
    --input input_video.mp4 \
    --output output_video.mp4 \
    --show-trails \
    --show-angles \
    --trail-length 30
```

### Parameters

- `--input`: Path to input climbing video
- `--output`: Path for output video with tracking
- `--show-trails`: Enable motion trails (shows movement history)
- `--show-angles`: Display joint angles
- `--trail-length`: Number of frames to show in motion trails (default: 30)

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

## Future Enhancements

- Real-time processing for live climbing sessions
- Route analysis and hold sequence tracking
- Performance comparison between multiple attempts
- Integration with climbing apps and databases
