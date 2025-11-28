from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Tuple

import cv2
import numpy as np

from core.llm import (
    ChatTextMessage,
    ChatVisionMessage,
    ChatCompleteParams,
    OllamaConnectionConfig,
    chat_complete,
    get_client,
    print_stats,
)

# Hard-coded IO paths (non-configurable, as requested)
INPUT_VIDEO_PATH = "./workspace/input.mp4"
OUTPUT_VIDEO_PATH = "./workspace/output.mp4"
FRAMES_JSONL_PATH = "./workspace/frames.jsonl"
FRAMES_DEBUG_DIR = "./workspace/frames"

# Model variant choice (centralized to avoid duplication)
CURRENT_MODEL_NAME = "qwen3-vl:32b-thinking"

# Qwen3-VL constraints / heuristics:
# - Aspect ratio long/short must be <= 200 (per official docs).
# - Width & height must be > 10 pixels.
# - For grounding, the model uses a normalized coordinate system scaled to [0, 1000].
# - The Qwen3-VL image preprocessor's pixel budget for a single image (H x W) is:
#     * min_pixels  = 65,536   (≈ 256 x 256)
#     * max_pixels  = 16,777,216 (≈ 4096 x 4096)
#   In the official Qwen3-VL processors, these are enforced on the *total* number
#   of pixels H * W (not per-side caps), with aspect-ratio constraints handled
#   separately. We replicate that logic here as a hard validation gate and refuse
#   to resize frames ourselves; any dynamic resizing & patch-aligned handling are
#   delegated to Ollama's Qwen3-VL image processor.
MAX_ABS_ASPECT_RATIO = 200.0
MIN_DIMENSION = 10
QWEN_NORMALIZATION_GRID = 1000.0  # coordinates are in [0, 1000]

QWEN3_VL_MIN_PIXELS = 65536        # 256 x 256, from Qwen3-VL preprocessor_config shortest_edge
QWEN3_VL_MAX_PIXELS = 16777216     # 4096 x 4096, from Qwen3-VL preprocessor_config longest_edge

# Palette of dark, high-contrast BGR colors used to style bounding boxes and label
# backgrounds based on the *index* of the label in the user-provided list.
# The first entry preserves the original red styling for the first label.
LABEL_COLOR_PALETTE_BGR: List[Tuple[int, int, int]] = [
    (0, 0, 255),      # red (first label)
    (0, 128, 0),      # dark green
    (255, 0, 0),      # dark blue
    (0, 128, 128),    # dark cyan / teal
    (128, 0, 128),    # dark magenta / purple
    (128, 128, 0),    # olive
    (128, 0, 0),      # maroon
    (0, 128, 255),    # dark orange
]


@dataclass
class NormalizedBBox:
    """
    Bounding box in Qwen3-VL's normalized coordinate space.
    """
    label: str
    x1: int
    y1: int
    x2: int
    y2: int


def probe_video_metadata(path: str) -> tuple[int, int, float, float]:
    """
    Inspect the input video using ffprobe and return (width, height, fps, duration_seconds).

    Any structural issues or missing fields are treated as fatal.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,duration:format=duration",
        "-of",
        "json",
        path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe executable not found. Ensure FFmpeg (including ffprobe) is installed "
            "and available on PATH."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            "ffprobe failed to inspect the input video.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {result.returncode}\n"
            f"Stderr:\n{result.stderr}"
        )

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "ffprobe returned non-JSON output when JSON was requested.\n"
            f"Raw stdout (first 1000 chars): {result.stdout[:1000]!r}"
        ) from exc

    streams = info.get("streams")
    if not streams:
        raise RuntimeError(
            "ffprobe did not return any video streams for the input file.\n"
            f"Command: {' '.join(cmd)}"
        )

    stream = streams[0]

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(
            "ffprobe reported non-positive width/height for the input video.\n"
            f"Reported width={width}, height={height}."
        )

    # FPS extraction is now strict: either we get a sane, positive FPS value or we fail loudly.
    raw_avg_frame_rate = stream.get("avg_frame_rate")

    def _stream_meta_snippet() -> str:
        try:
            return json.dumps(stream, indent=2)[:1000]
        except Exception:
            return repr(stream)[:1000]

    if raw_avg_frame_rate is None:
        raise RuntimeError(
            "ffprobe did not report 'avg_frame_rate' for the primary video stream. "
            "Expected a rational string like '30000/1001'.\n"
            f"Stream metadata snippet:\n{_stream_meta_snippet()}"
        )

    if not isinstance(raw_avg_frame_rate, str):
        raise RuntimeError(
            "ffprobe reported 'avg_frame_rate' with an unexpected type.\n"
            f"Expected a string like '30000/1001', got {type(raw_avg_frame_rate).__name__}: "
            f"{raw_avg_frame_rate!r}.\n"
            f"Stream metadata snippet:\n{_stream_meta_snippet()}"
        )

    if "/" not in raw_avg_frame_rate:
        raise RuntimeError(
            "ffprobe reported 'avg_frame_rate' in an unexpected format.\n"
            f"Expected '<numerator>/<denominator>', got {raw_avg_frame_rate!r}.\n"
            f"Stream metadata snippet:\n{_stream_meta_snippet()}"
        )

    try:
        fps_fraction = Fraction(raw_avg_frame_rate)
    except ZeroDivisionError as exc:
        raise RuntimeError(
            "ffprobe reported 'avg_frame_rate' with a zero denominator, which is invalid.\n"
            f"avg_frame_rate={raw_avg_frame_rate!r}.\n"
            f"Stream metadata snippet:\n{_stream_meta_snippet()}"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            "ffprobe reported 'avg_frame_rate' that could not be parsed as a rational number.\n"
            f"avg_frame_rate={raw_avg_frame_rate!r}.\n"
            f"Error: {exc}\n"
            f"Stream metadata snippet:\n{_stream_meta_snippet()}"
        ) from exc

    fps = float(fps_fraction)

    if not math.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(
            "Derived FPS from ffprobe 'avg_frame_rate' is non-positive or non-finite.\n"
            f"avg_frame_rate={raw_avg_frame_rate!r}, derived fps={fps!r}.\n"
            f"Stream metadata snippet:\n{_stream_meta_snippet()}"
        )

    # Sanity check: reject extremely small or extremely large FPS values as likely metadata bugs.
    # Typical real-world frame rates are in the ~1–240 fps range; we allow a wider band but still
    # insist on something that looks sane.
    if fps < 1.0 or fps > 1000.0:
        raise RuntimeError(
            "Derived FPS from ffprobe 'avg_frame_rate' is outside the expected sanity range "
            "[1.0, 1000.0]. This likely indicates problematic or missing metadata.\n"
            f"avg_frame_rate={raw_avg_frame_rate!r}, derived fps={fps:.6f}.\n"
            "If this FPS is actually correct for your content, adjust the sanity check in "
            "probe_video_metadata().\n"
            f"Stream metadata snippet:\n{_stream_meta_snippet()}"
        )

    # Prefer stream-level duration, but fall back to format-level if needed.
    duration = 0.0
    duration_raw = stream.get("duration")
    if duration_raw is None:
        fmt = info.get("format")
        if isinstance(fmt, dict):
            duration_raw = fmt.get("duration")
    if duration_raw is not None:
        try:
            duration = float(duration_raw)
        except (TypeError, ValueError):
            duration = 0.0

    return width, height, fps, duration


class FFmpegDecodeStream:
    """
    Thin wrapper around an ffmpeg CLI process that decodes a video file into raw BGR frames.
    """

    def __init__(
        self,
        input_path: str,
        width: int,
        height: int,
        max_frames: int | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(
                f"FFmpegDecodeStream requires positive width/height, got {width}x{height}."
            )

        self.input_path = input_path
        self.width = width
        self.height = height
        self.frame_size_bytes = width * height * 3
        self._closed = False

        cmd: list[str] = [
            "ffmpeg",
            "-v",
            "info",
            "-i",
            input_path,
        ]
        if max_frames is not None:
            if max_frames <= 0:
                raise ValueError(
                    f"FFmpegDecodeStream max_frames must be positive when provided, got {max_frames}."
                )
            cmd += ["-vframes", str(max_frames)]
        cmd += [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        self.cmd = cmd

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=None,  # Inherit stderr so ffmpeg logs are visible in real time.
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Failed to start ffmpeg for decoding input video: executable not found. "
                "Ensure 'ffmpeg' is installed and available on PATH."
            ) from exc

        if proc.stdout is None:
            raise RuntimeError(
                "Internal error: ffmpeg decode process was started without a stdout pipe."
            )

        self.proc = proc
        self.stdout = proc.stdout

    def _read_exact(self, n: int) -> bytes | None:
        """
        Read exactly n bytes from ffmpeg's stdout, or return None if EOF is reached
        before any bytes are read.
        """
        chunks: list[bytes] = []
        bytes_read = 0

        while bytes_read < n:
            chunk = self.stdout.read(n - bytes_read)
            if not chunk:
                if bytes_read == 0:
                    return None
                raise RuntimeError(
                    "ffmpeg decode process ended before a full frame could be read.\n"
                    f"Expected {n} bytes, got {bytes_read} bytes.\n"
                    f"Command: {' '.join(self.cmd)}"
                )
            chunks.append(chunk)
            bytes_read += len(chunk)

        return b"".join(chunks)

    def read_frame(self) -> np.ndarray | None:
        """
        Read a single frame as a uint8 BGR numpy array with shape (height, width, 3).

        Returns None on clean EOF.
        """
        if self._closed:
            return None

        buf = self._read_exact(self.frame_size_bytes)
        if buf is None:
            return None

        # Create an array from the raw buffer and ensure it is writeable so OpenCV
        # can draw on it in-place during rendering.
        frame = np.frombuffer(buf, dtype=np.uint8)
        try:
            frame = frame.reshape((self.height, self.width, 3))
        except ValueError as exc:
            raise RuntimeError(
                "Decoded raw frame from ffmpeg has unexpected size when reshaping.\n"
                f"Expected {self.height}x{self.width}x3 bytes."
            ) from exc

        if not frame.flags.writeable:
            frame = frame.copy()

        return frame

    def _wait(self, expect_zero_exit: bool) -> None:
        if self._closed:
            return
        ret = self.proc.wait()
        self._closed = True
        if expect_zero_exit and ret != 0:
            raise RuntimeError(
                "ffmpeg decode process exited with a non-zero status.\n"
                f"Command: {' '.join(self.cmd)}\n"
                f"Exit code: {ret}"
            )

    def close(self, expect_zero_exit: bool = True) -> None:
        if self._closed:
            return
        try:
            if self.stdout and not self.stdout.closed:
                self.stdout.close()
        finally:
            self._wait(expect_zero_exit=expect_zero_exit)

    def terminate(self) -> None:
        if self._closed:
            return
        try:
            self.proc.terminate()
        finally:
            try:
                if self.stdout and not self.stdout.closed:
                    self.stdout.close()
            finally:
                # We explicitly do not enforce a zero exit code when terminating.
                self._wait(expect_zero_exit=False)


class FFmpegEncodeStream:
    """
    Thin wrapper around an ffmpeg CLI process that encodes raw BGR frames into MP4.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(
                f"FFmpegEncodeStream requires positive width/height, got {width}x{height}."
            )
        if fps <= 0:
            raise ValueError(
                f"FFmpegEncodeStream requires a positive fps value, got {fps!r}."
            )

        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self._closed = False

        # Opinionated choice: always encode as MP4 using H.264 (libx264) with a high-quality
        # CRF-based setting that keeps 4K looking clean while still using sane compression.
        cmd: list[str] = [
            "ffmpeg",
            "-v",
            "info",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-y",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",      # better compression efficiency for a given quality
            "-crf",
            "17",        # visually near-lossless for 4K in most cases
            "-pix_fmt",
            "yuv420p",   # widely compatible
            output_path,
        ]
        self.cmd = cmd

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=None,  # Inherit stderr so ffmpeg logs are visible in real time.
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Failed to start ffmpeg for encoding output video: executable not found. "
                "Ensure 'ffmpeg' with libx264 support is installed and available on PATH."
            ) from exc

        if proc.stdin is None:
            raise RuntimeError(
                "Internal error: ffmpeg encode process was started without a stdin pipe."
            )

        self.proc = proc
        self.stdin = proc.stdin

    def write_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Encode a single uint8 BGR frame with shape (height, width, 3).
        """
        if self._closed:
            raise RuntimeError(
                "Attempted to write a frame to a closed ffmpeg encode stream."
            )

        h, w = frame_bgr.shape[:2]
        if (w, h) != (self.width, self.height):
            raise RuntimeError(
                "Attempted to encode a frame with unexpected resolution.\n"
                f"Expected {self.width}x{self.height}, got {w}x{h}."
            )

        if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise RuntimeError(
                "Frames passed to FFmpegEncodeStream.write_frame must be uint8 BGR images "
                "with shape (height, width, 3)."
            )

        try:
            self.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(
                "ffmpeg encode process closed its input pipe unexpectedly while writing a frame.\n"
                f"Command: {' '.join(self.cmd)}"
            ) from exc

    def _wait(self, expect_zero_exit: bool) -> None:
        if self._closed:
            return
        ret = self.proc.wait()
        self._closed = True
        if expect_zero_exit and ret != 0:
            raise RuntimeError(
                "ffmpeg encode process exited with a non-zero status.\n"
                f"Command: {' '.join(self.cmd)}\n"
                f"Exit code: {ret}\n"
                "This likely means your ffmpeg build does not provide an encoder for the "
                "explicitly requested 'libx264' codec, or the output file could not be written."
            )

    def close(self, expect_zero_exit: bool = True) -> None:
        if self._closed:
            return
        try:
            if self.stdin and not self.stdin.closed:
                self.stdin.close()
        finally:
            self._wait(expect_zero_exit=expect_zero_exit)

    def terminate(self) -> None:
        if self._closed:
            return
        try:
            try:
                if self.stdin and not self.stdin.closed:
                    self.stdin.close()
            finally:
                self.proc.terminate()
        finally:
            # We explicitly do not enforce a zero exit code when terminating.
            self._wait(expect_zero_exit=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline object tracking with Qwen3-VL on ./workspace/input.mp4.\n"
            "Configurable inputs are: --description (what to look for) and "
            "--label/--labels (the finite set of labels the model must choose from)."
        )
    )
    parser.add_argument(
        "--description",
        required=True,
        help=(
            "Natural language description of the objects to track. "
            "You can describe multiple object types here, e.g. "
            "'all cars and pedestrians wearing red jackets'."
        ),
    )

    label_group = parser.add_mutually_exclusive_group(required=True)
    label_group.add_argument(
        "--label",
        dest="label",
        action="append",
        help=(
            "Allowed object label. Can be specified multiple times. "
            "Each detected object must choose exactly one label from the allowed set."
        ),
    )
    label_group.add_argument(
        "--labels",
        dest="labels",
        help=(
            "Comma-separated list of allowed object labels. "
            "Each detected object must choose exactly one label from this list."
        ),
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Preview mode: use however many frames are present in the existing JSONL file "
            "to render an annotated video, then exit without running the LLM on any new frames."
        ),
    )

    parser.add_argument(
        "--boxes-only",
        dest="boxes_only",
        action="store_true",
        help=(
            "Draw only colored bounding boxes (no label text or background). "
            "Box color is determined by the index of the label in the allowed label list."
        ),
    )

    # No other CLI arguments are supported beyond --description, the labeling options,
    # --preview, and --boxes-only, by design.
    args = parser.parse_args()

    # Normalize the labels into a single canonical list for downstream code.
    allowed_labels: List[str]
    if args.label:
        allowed_labels = [lbl.strip() for lbl in args.label if lbl and lbl.strip()]
    else:
        parts = (args.labels or "").split(",")
        allowed_labels = [part.strip() for part in parts if part.strip()]

    if not allowed_labels:
        parser.error(
            "At least one non-empty label must be provided via --label or --labels."
        )

    # Attach normalized labels to the args namespace.
    args.allowed_labels = allowed_labels

    return args


def lawyerly_video_checks(width: int, height: int) -> None:
    """
    Enforce Qwen3-VL-style constraints in an opinionated, 'lawyerly' way.

    This function unifies:
      - Basic sanity checks on the frame geometry, and
      - The Qwen3-VL image budget constraints derived from its official
        preprocessor_config.json.

    Constraints enforced (all must pass, or we raise a detailed ValueError):

    1. Positive dimensions:
       - width > 0 and height > 0 are required for any sensible image.

    2. Minimum dimension:
       - width >= MIN_DIMENSION and height >= MIN_DIMENSION.
       - Qwen3-VL docs require width & height > 10 pixels; we keep the same
         lawyerly lower bound to avoid absurdly tiny frames.

    3. Aspect ratio:
       - max(width, height) / min(width, height) <= MAX_ABS_ASPECT_RATIO.
       - Qwen3-VL specifies that the absolute aspect ratio must not exceed 200:1.

    4. Total pixel budget (H * W):
       - QWEN3_VL_MIN_PIXELS <= width * height <= QWEN3_VL_MAX_PIXELS.
       - The official Qwen3-VL preprocessor defines:
           * shortest_edge = 65,536
           * longest_edge  = 16,777,216
         and uses these as bounds on H * W (total number of pixels) rather
         than independent per-side caps. That means:
           * Very small images (< ~256 x 256) are considered out-of-budget.
           * Very large images (> ~4096 x 4096 in area) must be downscaled.
         We do *not* downscale or upscale here; if a frame falls outside this
         range, we fail loudly and ask the caller to adjust the video upstream.

    Note: Qwen3-VL's own image processor will apply its dynamic resizing /
    patch-aligned logic internally (SmartResize / multiples of 32, etc.).
    Our job in this function is to ensure that every frame is at least in
    the same broad regime as the model's training-time preprocessing.
    """
    errors: list[str] = []

    # 1. Positive dimensions.
    if width <= 0 or height <= 0:
        errors.append(
            "Video has non-positive dimensions.\n"
            f"  - width={width}, height={height}\n"
            "Both width and height must be strictly positive for Qwen3-VL."
        )

    # 2. Minimum dimension.
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        errors.append(
            "Video frame dimensions are too small for Qwen3-VL.\n"
            f"  - width x height = {width} x {height}\n"
            f"  - required: width >= {MIN_DIMENSION} and height >= {MIN_DIMENSION}\n"
            "Qwen3-VL requires width and height to be greater than 10 pixels."
        )

    # 3. Aspect ratio.
    if width > 0 and height > 0:
        long_edge = max(width, height)
        short_edge = min(width, height)
        if short_edge > 0:
            aspect_ratio = long_edge / short_edge
            if aspect_ratio > MAX_ABS_ASPECT_RATIO:
                errors.append(
                    "No, you can't use that video: its aspect ratio exceeds Qwen3-VL's "
                    "recommended limit.\n"
                    f"  - width x height = {width} x {height}\n"
                    f"  - long/short edge ratio = {aspect_ratio:.2f} > {MAX_ABS_ASPECT_RATIO:.0f}\n"
                    "The absolute aspect ratio must be <= 200:1 or 1:200 according to the Qwen-VL docs."
                )

    # 4. Total pixel budget.
    if width > 0 and height > 0:
        pixels = width * height
        if pixels < QWEN3_VL_MIN_PIXELS:
            errors.append(
                "Frame is too small for Qwen3-VL's configured pixel budget.\n"
                f"  - width x height = {width} x {height} = {pixels} pixels\n"
                f"  - required minimum total pixels (H*W) = {QWEN3_VL_MIN_PIXELS} "
                "(≈ 256 x 256)\n"
                "This script refuses to upscale tiny frames automatically. "
                "Use a higher-resolution input video or pre-process it upstream."
            )
        if pixels > QWEN3_VL_MAX_PIXELS:
            errors.append(
                "Frame is too large for Qwen3-VL's configured pixel budget.\n"
                f"  - width x height = {width} x {height} = {pixels} pixels\n"
                f"  - allowed maximum total pixels (H*W) = {QWEN3_VL_MAX_PIXELS} "
                "(≈ 4096 x 4096)\n"
                "This script refuses to downscale oversized frames automatically. "
                "Downscale the video externally (e.g. via ffmpeg) before using this script."
            )

    if errors:
        raise ValueError(
            "Video frame does not satisfy Qwen3-VL's geometry / pixel-budget constraints:\n\n"
            + "\n\n".join(errors)
        )


def frame_to_base64_png(frame_bgr: np.ndarray) -> str:
    """
    Encode a frame as base64 PNG for Qwen3-VL.

    We deliberately do NOT resize here. The frame is passed at the original
    video resolution (subject to the unified Qwen3-VL lawyerly checks), and the
    Qwen3-VL image processor inside Ollama is responsible for any dynamic
    resizing / patch-aligned handling.

    PNG is chosen to avoid introducing an extra layer of lossy JPEG compression
    before Ollama decodes the image and applies Qwen's preprocessing. This keeps
    the pixel content as faithful as possible to the decoded video frame.
    """
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            "frame_to_base64_png expects a uint8 BGR image with shape (height, width, 3). "
            f"Got dtype={frame_bgr.dtype}, shape={frame_bgr.shape!r}."
        )

    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to PNG-encode frame for LLM input.")
    return base64.b64encode(buf).decode("ascii")


def save_debug_frame_from_base64_image(image_b64: str, frame_index: int) -> None:
    """
    Persist exactly what is being sent to the LLM for debugging.

    Behavior (opinionated, with no silent fallbacks):
    - Decode the base64 string that is fed into the LLM.
    - Write the raw bytes directly as a PNG file to
      ./workspace/frames/0000000001.png (and so on), where the number is
      the 1-based frame index in the original video, zero-padded to 10 digits.
      This preserves the exact compressed representation the LLM sees.
    - If base64 decoding or PNG writing fails, raise a verbose RuntimeError
      describing what was supposed to happen and what actually failed.
    """
    # Ensure the debug directory exists.
    try:
        os.makedirs(FRAMES_DEBUG_DIR, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to create the debug frames directory for persisting LLM input images.\n"
            f"Expected to create or reuse directory {FRAMES_DEBUG_DIR!r}, but os.makedirs "
            f"raised {type(exc).__name__}: {exc}"
        ) from exc

    frame_number = frame_index + 1
    png_path = os.path.join(FRAMES_DEBUG_DIR, f"{frame_number:010d}.png")

    # Decode the base64 payload that is being sent to the LLM.
    try:
        img_bytes = base64.b64decode(image_b64.encode("ascii"), validate=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to base64-decode the LLM image payload while attempting to write a "
            "debug frame file.\n"
            f"This should have been a valid ASCII base64 string representing the exact "
            f"bytes fed into the LLM for frame_index={frame_index}, but decoding raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # Persist the exact compressed representation as a PNG file on disk.
    try:
        with open(png_path, "wb") as f:
            f.write(img_bytes)
    except Exception as exc:
        raise RuntimeError(
            "Failed to persist a debug PNG image file representing the exact content fed "
            "into the LLM for a given frame.\n"
            f"- Operation (writing raw PNG bytes to {png_path!r}) failed with "
            f"{type(exc).__name__}: {exc}\n"
            "As a result, no debug image file could be written for this frame. This "
            "indicates an unexpected filesystem or data integrity issue."
        ) from exc


def strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> blocks from Qwen3-VL thinking outputs so we can parse JSON.
    """
    lower = text.lower()
    start_token = "<think>"
    end_token = "</think>"

    start_idx = lower.find(start_token)
    if start_idx == -1:
        return text

    end_idx = lower.find(end_token, start_idx + len(start_token))
    if end_idx == -1:
        # Drop everything from <think> onward.
        return text[:start_idx]

    # Remove the think block.
    return text[:start_idx] + text[end_idx + len(end_token) :]


def extract_json_substring(raw: str) -> str:
    """
    Try to extract a pure JSON payload (ideally a list) from the model output.

    Strategy:
    1. Strip <think> tags.
    2. First try to parse the whole string as JSON.
    3. If that fails, look for the first '[' and last ']' and parse the slice in between.
    """
    stripped = strip_think_tags(raw).strip()

    # Quick path: maybe it's pure JSON already.
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return stripped
        except json.JSONDecodeError:
            pass

    # Fallback: try to locate the outermost JSON array.
    left = stripped.find("[")
    right = stripped.rfind("]")
    if left == -1 or right == -1 or left >= right:
        raise ValueError(
            f"Could not locate a JSON array in model output. "
            f"First 200 chars: {stripped[:200]!r}"
        )

    candidate = stripped[left : right + 1]
    # Validate that this substring is at least syntactically valid JSON.
    json.loads(candidate)
    return candidate


def parse_bboxes_from_text(
    text: str,
    allowed_labels: List[str] | None = None,
) -> List[NormalizedBBox]:
    """
    Parse and validate the model output into a list of NormalizedBBox instances.

    Requirements:
    - Top-level JSON is a list (or a dict with a 'boxes' key that is a list).
    - Each entry is an object with:
        - 'bbox_2d' (preferred) or 'bbox' : list of 4 numbers [x1, y1, x2, y2]
        - 'label' (preferred) or 'category'/'text' : non-empty string
    - Each coordinate must be a finite number in [0, 1000].
    - Coordinates are coerced to ints; must satisfy x1 < x2 and y1 < y2.
    - If allowed_labels is provided, each 'label' must match (case-insensitive) one of them.
    """
    try:
        json_payload = extract_json_substring(text)
    except Exception as exc:
        raise ValueError(f"Failed to isolate JSON from model output: {exc}") from exc

    try:
        data = json.loads(json_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output is not valid JSON: {exc}") from exc

    # Accept either a plain list or a {"boxes": [...]} container.
    if isinstance(data, dict) and "boxes" in data:
        data = data["boxes"]

    if not isinstance(data, list):
        raise ValueError(f"Top-level JSON must be a list, got {type(data).__name__}.")

    boxes: List[NormalizedBBox] = []

    canonical_label_map: dict[str, str] | None = None
    if allowed_labels is not None:
        canonical_label_map = {
            lbl.strip().lower(): lbl.strip()
            for lbl in allowed_labels
            if lbl and lbl.strip()
        }
        if not canonical_label_map:
            raise ValueError("allowed_labels must contain at least one non-empty label.")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Entry {idx} in JSON is not an object; got {type(item).__name__}."
            )

        if "bbox_2d" in item:
            raw_bbox = item["bbox_2d"]
        elif "bbox" in item:
            raw_bbox = item["bbox"]
        else:
            raise ValueError(f"Entry {idx} is missing 'bbox_2d' (or 'bbox') field.")

        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            raise ValueError(
                f"Entry {idx} 'bbox_2d' must be a list of 4 numbers, got: {raw_bbox!r}"
            )

        coords_f: list[float] = []
        for j, v in enumerate(raw_bbox):
            try:
                f = float(v)
            except Exception:
                raise ValueError(
                    f"Entry {idx} coordinate {j} is not numeric: {v!r}"
                ) from None
            if not math.isfinite(f):
                raise ValueError(
                    f"Entry {idx} coordinate {j} is not finite: {f!r}"
                )
            if not (0.0 <= f <= 1000.0):
                raise ValueError(
                    f"Entry {idx} coordinate {j} is out of [0, 1000] range: {f!r}"
                )
            coords_f.append(f)

        x1_i, y1_i, x2_i, y2_i = [int(round(v)) for v in coords_f]

        if not (x1_i < x2_i and y1_i < y2_i):
            raise ValueError(
                f"Entry {idx} must satisfy x1 < x2 and y1 < y2, got: "
                f"[{x1_i}, {y1_i}, {x2_i}, {y2_i}]"
            )

        raw_label = item.get("label") or item.get("category") or item.get("text")
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError(
                f"Entry {idx} missing a non-empty 'label'/'category'/'text' field."
            )

        label = raw_label.strip()
        if canonical_label_map is not None:
            key = label.lower()
            if key not in canonical_label_map:
                raise ValueError(
                    f"Entry {idx} label {label!r} is not in the allowed label set: "
                    f"{sorted(canonical_label_map.values())!r}."
                )
            # Canonicalize to the user-provided spelling.
            label = canonical_label_map[key]

        boxes.append(
            NormalizedBBox(
                label=label,
                x1=x1_i,
                y1=y1_i,
                x2=x2_i,
                y2=y2_i,
            )
        )

    return boxes


def draw_bboxes_on_frame(
    frame_bgr: np.ndarray,
    bboxes: List[NormalizedBBox],
    allowed_labels: List[str],
    boxes_only: bool = False,
) -> None:
    """
    Draw normalized bounding boxes on the frame in-place at the original video resolution.

    Box and label background colors are chosen based on the index of the label in the
    user-specified allowed_labels list (with wrap-around over a small, high-quality
    color palette). Text is always rendered in white. When boxes_only=True, only the
    colored boxes are drawn (no label text or background).
    """
    height, width = frame_bgr.shape[:2]

    if not bboxes:
        return

    # Dynamic styling for readability.
    min_side = min(width, height)
    thickness = max(2, int(round(min_side / 400)))
    font_scale = max(0.5, min_side / 800.0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Precompute a stable mapping from canonical label -> index, then to BGR color.
    label_to_index: dict[str, int] = {}
    for idx, lbl in enumerate(allowed_labels):
        canonical = lbl.strip()
        if not canonical:
            continue
        if canonical not in label_to_index:
            label_to_index[canonical] = idx

    def _color_for_label(label: str) -> Tuple[int, int, int]:
        idx = label_to_index.get(label, 0)
        palette_idx = idx % len(LABEL_COLOR_PALETTE_BGR)
        return LABEL_COLOR_PALETTE_BGR[palette_idx]

    color_text = (255, 255, 255)

    for box in bboxes:
        # Map from [0, 1000] grid to pixel coordinates.
        x1 = int(round((box.x1 / QWEN_NORMALIZATION_GRID) * (width - 1)))
        y1 = int(round((box.y1 / QWEN_NORMALIZATION_GRID) * (height - 1)))
        x2 = int(round((box.x2 / QWEN_NORMALIZATION_GRID) * (width - 1)))
        y2 = int(round((box.y2 / QWEN_NORMALIZATION_GRID) * (height - 1)))

        # Clamp to image bounds.
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x1 >= x2 or y1 >= y2:
            # Degenerate after rounding; skip this one.
            continue

        # Box & label background color: chosen from the palette based on label index;
        # text is always white for contrast.
        color_box = _color_for_label(box.label)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_box, thickness)

        if boxes_only:
            # In boxes-only mode we deliberately skip drawing label text and its background
            # to hide underlying video content as little as possible.
            continue

        label_text = box.label
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )
        text_x = x1
        text_y = max(y1 - 4, text_h + 2)

        # Filled rectangle behind text for readability.
        cv2.rectangle(
            frame_bgr,
            (text_x, text_y - text_h - baseline),
            (text_x + text_w, text_y + baseline),
            color_box,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            frame_bgr,
            label_text,
            (text_x, text_y),
            font,
            font_scale,
            color_text,
            thickness,
            lineType=cv2.LINE_AA,
        )


def build_llm_messages(
    image_b64: str,
    description: str,
    allowed_labels: List[str],
) -> list:
    """
    Build the message list for Qwen3-VL, closely following Alibaba's grounding style.

    In addition to the description, the model is given a finite set of valid labels.
    For each detected object, it must choose exactly one label from this set.
    """
    labels_block = "\n".join(f"- {label}" for label in allowed_labels)

    system_prompt = (
        "You are a precise 2D grounding assistant based on Qwen3-VL.\n"
        "Given **one image**, you must locate the requested objects and return their "
        "2D bounding boxes ONLY in JSON format.\n\n"
        "Bounding boxes must follow this schema (array of objects):\n"
        "[\n"
        '  {"label": "<short object label>", "bbox_2d": [x1, y1, x2, y2]},\n'
        "  ...\n"
        "]\n\n"
        "Valid labels (you MUST choose from this list and may not invent new labels):\n"
        f"{labels_block}\n\n"
        "Rules:\n"
        "- Coordinates are normalized on a 1000x1000 grid.\n"
        "- Each coordinate is an integer in [0, 1000].\n"
        "- x1 < x2 and y1 < y2.\n"
        "- For each detected object, the 'label' field must be EXACTLY one of the valid\n"
        "  labels listed above (match the spelling as closely as possible).\n"
        "- If no matching object exists, return [].\n"
        "- Do NOT output any extra text, comments, or explanations. Only pure JSON."
        "- Exhaustively and thoroughly create a **complete** list or everything relevant that is visible with one unique bounding box for each object, even when very far away or taking a tiny portion of the frame."
    )

    user_prompt = (
        "Locate every object that matches the following description in the image and "
        "return their 2D bounding boxes using the required JSON schema.\n\n"
        f"User instruction: {description}\n"
        "I need the list to be very much exhaustive, including even the smallest, farthest away, and least visible objects on screen, individually bounded.\n\n"
        "You are also given a finite set of valid labels. For each detected object you\n"
        "must choose exactly one label from this set and use it as the 'label' value in\n"
        "the JSON output. Do not invent any new labels or synonyms.\n\n"
        "Valid labels:\n"
        f"{labels_block}\n\n"
        "Remember: output ONLY the JSON array. If there are no matches, output []."
    )

    messages = [
        ChatTextMessage(role="system", content=system_prompt),
        ChatVisionMessage(role="user", content=user_prompt, images_b64=[image_b64]),
    ]
    return messages


def run_llm_for_frame(
    frame_bgr: np.ndarray,
    description: str,
    allowed_labels: List[str],
    frame_index: int,
    fps: float,
    duration: float,
    client,
    connection: OllamaConnectionConfig,
) -> List[NormalizedBBox]:
    """
    Run Qwen3-VL on a single frame, retrying until we get sane bounding boxes.

    The loop has *no* fallback other than asking the LLM again; any malformed
    output leads to another attempt, with the error printed for visibility.

    Progress reporting is based on timestamps derived from fps/duration rather than
    total frame counts, so we never rely on nb_frames metadata.
    """
    if fps <= 0.0 or not math.isfinite(fps):
        raise ValueError(
            "run_llm_for_frame expected a positive, finite fps value. "
            f"Got fps={fps!r}. This indicates a bug in probe_video_metadata() or an "
            "invalid caller."
        )

    attempt = 0
    model_name = CURRENT_MODEL_NAME

    def _progress_prefix() -> str:
        """
        Build a human-friendly progress prefix using time-based information.
        """
        display_frame = frame_index + 1

        # fps is guaranteed to be positive and finite by the check at the start of
        # this function.
        current_time_s = frame_index / fps
        if duration > 0.0:
            progress = min(1.0, current_time_s / duration)
            return (
                f"[frame {display_frame}]"
                f"[t={current_time_s:.2f}s/{duration:.2f}s "
                f"({progress * 100.0:5.1f}%)]"
            )
        else:
            return f"[frame {display_frame}][t={current_time_s:.2f}s]"

    while True:
        attempt += 1

        # NOTE: This is the exact point where we hand off vision preprocessing to Ollama.
        #
        # We deliberately do NOT implement any Qwen3-VL-specific resizing, pixel-budget
        # enforcement, or patch-aligned image handling here beyond the upfront checks in
        # lawyerly_video_checks() (which run once per video resolution). Instead, we:
        #
        #   1. Encode the raw decoded video frame as a PNG (frame_to_base64_png),
        #      preserving its original resolution as long as it satisfies Qwen3-VL's
        #      H*W pixel budget and aspect-ratio constraints.
        #   2. Send that base64 PNG to Ollama's qwen3-vl backend via the "images"
        #      field in the chat message.
        #
        # On the Ollama side, the qwen3vl image processor is responsible for:
        #   - Decoding the PNG bytes into a pixel tensor.
        #   - Applying the same dynamic-resolution "SmartResize" logic used by the
        #     official Qwen3-VL image processor, which:
        #       * enforces the model's min/max pixel budget on the decoded tensor;
        #       * ensures height/width are multiples of the internal patch size
        #         (e.g. 16, with merging as configured);
        #       * respects aspect ratio constraints.
        #   - Normalizing and packing the image into the Qwen3-VL vision encoder.
        #
        # By validating that the frame lies within Qwen3-VL's allowed pixel budget
        # and aspect-ratio regime before this call, and then passing the untouched
        # frame pixels to Ollama, we align our inference-time preprocessing with the
        # model's training-time assumptions without adding our own lossy downscaling
        # or ad-hoc resizing layer.
        b64_image = frame_to_base64_png(frame_bgr)

        # Persist a debug frame file constructed directly from the exact base64 payload
        # that is about to be sent to the LLM. This ensures that the PNG on disk
        # faithfully represents what the model actually receives.
        save_debug_frame_from_base64_image(b64_image, frame_index)

        messages = build_llm_messages(b64_image, description, allowed_labels)

        params = ChatCompleteParams(
            messages=messages,
            model=model_name,
            client=client,
            connection=connection,
            max_completion_tokens=8192,
            please_no_thinking=False,
            require_json=False,  # thinking models cannot use strict JSON mode
        )

        try:
            response = chat_complete(params)
        except Exception as exc:
            # Non-LLM issues (network, Ollama down) are fatal; we don't silently loop them.
            raise RuntimeError(
                f"Ollama / Qwen3-VL call failed on frame {frame_index + 1}, "
                f"attempt {attempt}: {exc}"
            ) from exc

        stats = print_stats(response)
        prefix = f"{_progress_prefix()}[attempt {attempt}]"
        if stats is not None:
            print(
                f"{prefix} LLM stats:\n{stats}",
                flush=True,
            )
        else:
            print(
                f"{prefix} LLM did not return detailed token-level stats.",
                flush=True,
            )

        raw_content = response.message.content

        try:
            boxes = parse_bboxes_from_text(raw_content, allowed_labels=allowed_labels)
        except Exception as exc:
            # The LLM misbehaved (non-JSON, bad coordinates, wrong labels, etc.); log and retry.
            preview = strip_think_tags(raw_content).strip()
            if len(preview) > 400:
                preview = preview[:400] + "...[truncated]"
            print(
                f"{prefix} LLM output invalid, will retry:\n"
                f"  error: {exc}\n"
                f"  raw (sanitized) preview: {preview}",
                flush=True,
            )
            continue

        print(
            f"{prefix} Successfully parsed {len(boxes)} bounding box(es).",
            flush=True,
        )
        return boxes


def parse_jsonl_line_to_bboxes(
    line: str,
    line_number: int,
    expected_frame_index: int,
    allowed_labels: List[str],
) -> List[NormalizedBBox]:
    """
    Parse and validate a single JSONL line into a list of NormalizedBBox instances.

    Each line must be a JSON object:
    {
      "frame_index": <int>,
      "boxes": [
        {"label": <str>, "x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>},
        ...
      ]
    }
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSONL line {line_number} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(obj, dict):
        raise ValueError(
            f"JSONL line {line_number} must be a JSON object, got {type(obj).__name__}."
        )

    if "frame_index" not in obj:
        raise ValueError(
            f"JSONL line {line_number} is missing required 'frame_index' field."
        )
    if "boxes" not in obj:
        raise ValueError(
            f"JSONL line {line_number} is missing required 'boxes' field."
        )

    frame_index = obj["frame_index"]
    if not isinstance(frame_index, int):
        raise ValueError(
            f"JSONL line {line_number} 'frame_index' must be an integer, "
            f"got {type(frame_index).__name__}."
        )
    if frame_index != expected_frame_index:
        raise ValueError(
            "JSONL file is out of sync with the input video.\n"
            f"Expected frame_index {expected_frame_index} at line {line_number}, "
            f"but found {frame_index}."
        )

    boxes_raw = obj["boxes"]
    if not isinstance(boxes_raw, list):
        raise ValueError(
            f"JSONL line {line_number} 'boxes' must be a list, "
            f"got {type(boxes_raw).__name__}."
        )

    canonical_label_map: dict[str, str] = {
        lbl.strip().lower(): lbl.strip()
        for lbl in allowed_labels
        if lbl and lbl.strip()
    }
    if not canonical_label_map:
        raise ValueError(
            "allowed_labels must contain at least one non-empty label when parsing JSONL."
        )

    boxes: List[NormalizedBBox] = []
    for idx, item in enumerate(boxes_raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"JSONL line {line_number} 'boxes[{idx}]' must be an object, "
                f"got {type(item).__name__}."
            )

        raw_label = item.get("label")
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError(
                f"JSONL line {line_number} 'boxes[{idx}]' is missing a non-empty 'label' field."
            )
        label = raw_label.strip()
        key = label.lower()
        if key not in canonical_label_map:
            raise ValueError(
                f"JSONL line {line_number} 'boxes[{idx}]' label {label!r} is not in the "
                f"allowed label set: {sorted(canonical_label_map.values())!r}."
            )
        label = canonical_label_map[key]

        for coord_name in ("x1", "y1", "x2", "y2"):
            if coord_name not in item:
                raise ValueError(
                    f"JSONL line {line_number} 'boxes[{idx}]' is missing '{coord_name}' field."
                )
            if not isinstance(item[coord_name], int):
                raise ValueError(
                    f"JSONL line {line_number} 'boxes[{idx}].{coord_name}' must be an integer, "
                    f"got {type(item[coord_name]).__name__}."
                )

        x1_i = item["x1"]
        y1_i = item["y1"]
        x2_i = item["x2"]
        y2_i = item["y2"]

        for coord_name, value in (("x1", x1_i), ("y1", y1_i), ("x2", x2_i), ("y2", y2_i)):
            if not (0 <= value <= 1000):
                raise ValueError(
                    f"JSONL line {line_number} 'boxes[{idx}].{coord_name}'={value} "
                    "is out of [0, 1000] range."
                )

        if not (x1_i < x2_i and y1_i < y2_i):
            raise ValueError(
                "JSONL line {line_number} 'boxes[{idx}]' must satisfy x1 < x2 and y1 < y2, "
                f"got [{x1_i}, {y1_i}, {x2_i}, {y2_i}]."
            )

        boxes.append(
            NormalizedBBox(
                label=label,
                x1=x1_i,
                y1=y1_i,
                x2=x2_i,
                y2=y2_i,
            )
        )

    return boxes


def validate_and_count_jsonl(jsonl_path: str, allowed_labels: List[str]) -> int:
    """
    Validate the entire JSONL file and return the number of frames it contains.

    Any deviation from the expected format or contents results in a loud exception.
    """
    frames_seen = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(
                    f"JSONL file {jsonl_path!r} contains an empty line at line {line_number}."
                )
            _ = parse_jsonl_line_to_bboxes(
                line=line,
                line_number=line_number,
                expected_frame_index=frames_seen,
                allowed_labels=allowed_labels,
            )
            frames_seen += 1
    return frames_seen


def process_video(
    description: str,
    allowed_labels: List[str],
    preview: bool = False,
    boxes_only: bool = False,
) -> None:
    if not os.path.exists(INPUT_VIDEO_PATH):
        raise FileNotFoundError(
            f"Expected input video at {INPUT_VIDEO_PATH!r}, but it does not exist."
        )

    orig_width, orig_height, fps, duration = probe_video_metadata(INPUT_VIDEO_PATH)

    duration_str = f"{duration:.3f} sec" if duration > 0.0 else "unknown"
    print(
        f"[video] Loaded {INPUT_VIDEO_PATH}\n"
        f"        resolution: {orig_width}x{orig_height}\n"
        f"        fps:        {fps:.3f}\n"
        f"        duration:   {duration_str}",
        flush=True,
    )

    # Unified lawyerly checks + Qwen3-VL pixel budget validation (may raise and abort early).
    lawyerly_video_checks(orig_width, orig_height)

    print(
        "[video] Frame resolution passes Qwen3-VL geometry & budget checks and will be "
        "sent to Ollama without additional resizing.",
        flush=True,
    )

    frames_already_done = 0
    if os.path.exists(FRAMES_JSONL_PATH):
        print(
            f"[resume] Detected existing JSONL file at {FRAMES_JSONL_PATH!r}, validating...",
            flush=True,
        )
        frames_already_done = validate_and_count_jsonl(
            FRAMES_JSONL_PATH, allowed_labels
        )
        print(
            f"[resume] JSONL file contains {frames_already_done} frame(s).",
            flush=True,
        )
    else:
        if preview:
            raise FileNotFoundError(
                f"Expected JSONL file at {FRAMES_JSONL_PATH!r} for preview mode, but it does not exist. "
                "Run the script without --preview first to generate bounding boxes."
            )
        os.makedirs(os.path.dirname(FRAMES_JSONL_PATH), exist_ok=True)

    final_jsonl_frame_count = 0
    client = None
    connection = None

    try:
        if not preview:
            # If base_url is omitted, it defaults to http://127.0.0.1:11434
            connection = OllamaConnectionConfig(base_url="http://172.17.0.1:11434")
            client = get_client(connection)

            # Stage 1: run the LLM for all frames that do not yet have JSONL entries.
            frame_index = 0
            jsonl_mode = "a" if frames_already_done > 0 else "w"
            decode_stream = FFmpegDecodeStream(
                INPUT_VIDEO_PATH,
                orig_width,
                orig_height,
                max_frames=None,
            )
            try:
                with open(FRAMES_JSONL_PATH, jsonl_mode, encoding="utf-8") as jsonl_file:
                    try:
                        while True:
                            frame_bgr = decode_stream.read_frame()
                            if frame_bgr is None:
                                break  # End of video.
                            h, w = frame_bgr.shape[:2]
                            if (w, h) != (orig_width, orig_height):
                                raise RuntimeError(
                                    "Encountered a frame with different resolution within the same video. "
                                    f"Expected {orig_width}x{orig_height}, got {w}x{h} at frame {frame_index}."
                                )

                            if frame_index < frames_already_done:
                                # This frame was already processed in a previous run; skip LLM.
                                frame_index += 1
                                continue

                            boxes = run_llm_for_frame(
                                frame_bgr=frame_bgr,
                                description=description,
                                allowed_labels=allowed_labels,
                                frame_index=frame_index,
                                fps=fps,
                                duration=duration,
                                client=client,
                                connection=connection,
                            )

                            record = {
                                "frame_index": frame_index,
                                "boxes": [
                                    {
                                        "label": box.label,
                                        "x1": box.x1,
                                        "y1": box.y1,
                                        "x2": box.x2,
                                        "y2": box.y2,
                                    }
                                    for box in boxes
                                ],
                            }
                            jsonl_file.write(json.dumps(record, separators=(",", ":")) + "\n")
                            jsonl_file.flush()

                            frame_index += 1
                    except KeyboardInterrupt:
                        print(
                            "\n[interrupt] Caught CTRL+C. Partial JSONL results have been preserved "
                            f"in {FRAMES_JSONL_PATH!r} and will be used for resuming.",
                            flush=True,
                        )
                        decode_stream.terminate()
                        raise SystemExit(130)
            finally:
                # If we're unwinding from an exception, don't let a non-zero ffmpeg exit
                # (for example due to a broken pipe) hide the real error.
                in_exception = sys.exc_info()[0] is not None
                decode_stream.close(expect_zero_exit=not in_exception)

            actual_frames_seen = frame_index

            # Validate that the JSONL file cleanly covers all frames that were observed.
            final_jsonl_frame_count = validate_and_count_jsonl(
                FRAMES_JSONL_PATH, allowed_labels
            )
            if final_jsonl_frame_count != actual_frames_seen:
                raise RuntimeError(
                    "Internal inconsistency between video frames and JSONL entries after processing.\n"
                    f"  - frames seen while reading video in Stage 1: {actual_frames_seen}\n"
                    f"  - valid JSONL entries:                       {final_jsonl_frame_count}\n"
                    "The script refuses to proceed with rendering an inconsistent state."
                )
        else:
            # Preview mode: take however many frames the JSONL file has and render only those.
            final_jsonl_frame_count = frames_already_done
            if final_jsonl_frame_count == 0:
                raise RuntimeError(
                    f"JSONL file {FRAMES_JSONL_PATH!r} contains zero frame entries; nothing to preview."
                )

        # Stage 2: now that the JSONL file has a clean entry for every frame, build the video.
        os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

        # fps has already been validated as positive and finite in probe_video_metadata().
        fps_out = fps

        decode_stream2 = FFmpegDecodeStream(
            INPUT_VIDEO_PATH,
            orig_width,
            orig_height,
            max_frames=final_jsonl_frame_count,
        )
        encode_stream = FFmpegEncodeStream(
            OUTPUT_VIDEO_PATH,
            orig_width,
            orig_height,
            fps_out,
        )

        try:
            frame_index = 0
            with open(FRAMES_JSONL_PATH, "r", encoding="utf-8") as jsonl_file:
                for line_number, raw_line in enumerate(jsonl_file, start=1):
                    line = raw_line.strip()
                    if not line:
                        raise ValueError(
                            f"JSONL file {FRAMES_JSONL_PATH!r} contains an empty line at "
                            f"line {line_number} while rendering."
                        )

                    boxes = parse_jsonl_line_to_bboxes(
                        line=line,
                        line_number=line_number,
                        expected_frame_index=frame_index,
                        allowed_labels=allowed_labels,
                    )

                    frame_bgr = decode_stream2.read_frame()
                    if frame_bgr is None:
                        raise RuntimeError(
                            "Input video ended earlier than expected while building the output video.\n"
                            f"Expected at least {final_jsonl_frame_count} frames, but "
                            f"ffmpeg decode stream returned EOF at frame index {frame_index}."
                        )

                    h, w = frame_bgr.shape[:2]
                    if (w, h) != (orig_width, orig_height):
                        raise RuntimeError(
                            "Encountered a frame with different resolution while rendering. "
                            f"Expected {orig_width}x{orig_height}, got {w}x{h} at frame {frame_index}."
                        )

                    draw_bboxes_on_frame(
                        frame_bgr,
                        boxes,
                        allowed_labels=allowed_labels,
                        boxes_only=boxes_only,
                    )
                    encode_stream.write_frame(frame_bgr)

                    frame_index += 1

            print(
                f"[done] Processed {frame_index} frame(s). "
                f"Annotated video written to {OUTPUT_VIDEO_PATH!r}.",
                flush=True,
            )
        finally:
            # As in Stage 1, avoid letting ffmpeg shutdown errors (e.g. broken pipes when
            # we've already stopped reading/writing) mask the original Python exception.
            in_exception = sys.exc_info()[0] is not None
            decode_stream2.close(expect_zero_exit=not in_exception)
            encode_stream.close(expect_zero_exit=not in_exception)
    finally:
        if client is not None:
            client.close()


def main() -> None:
    args = parse_args()
    try:
        process_video(
            args.description,
            args.allowed_labels,
            preview=args.preview,
            boxes_only=args.boxes_only,
        )
    except Exception as exc:
        # Complain loudly and exit non-zero.
        print(f"[fatal] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
