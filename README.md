# Local Object Tracking

Local offline (not realtime) object tracking in video using `Qwen3-VL-32B` running on `ollama-linux-amd64_v0.12.10`.

That LLM knows how to output json bounding boxes normalized to 1000x1000 pixels, so we basically run the image understanding separately on each frame of the video.
