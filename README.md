# Local Object Tracking

Local offline (not realtime) object tracking in video using `Qwen3-VL-32B` running on `ollama-linux-amd64_v0.12.10`.

That LLM knows how to output json bounding boxes normalized to 1000x1000 pixels, so we basically run the image understanding separately on each frame of the video.

## Usage

1. Create [./workspace/input.mp4](./workspace/input.mp4)

2. Run:
    ```sh
    python3 bbox.py --description "Anything that vaguely lookslike a colorful pixelated cloudy blob throughout the 2d/3d visualization" --labels "cloudy blob"
    ```

3. Wait minutes to hours, and get your [./workspace/input.mp4](./workspace/output.mp4)
