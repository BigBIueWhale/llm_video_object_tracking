from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import subprocess
from dataclasses import dataclass
from typing import List

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

# Qwen3-VL constraints / heuristics:
# - Aspect ratio long/short must be <= 200 (per official docs).
# - Width & height must be > 10 pixels.
# - For grounding, the model uses a 1000x1000 normalized grid for bbox coordinates.
MAX_ABS_ASPECT_RATIO = 200.0
MIN_DIMENSION = 10
QWEN_NORMALIZATION_GRID = 999.0  # coordinates are in [0, 999]


@dataclass
class NormalizedBBox:
    """
    Bounding box in Qwen3-VL's 1000×1000 normalized coordinate space.
    """
    label: str
    x1: int
    y1: int
    x2: int
    y2: int


def probe_video_metadata(path: str) -> tuple[int, int, float, int]:
    """
    Inspect the input video using ffprobe and return (width, height, fps, total_frames).

    Any structural issues or missing fields are treated as fatal.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
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

    avg_frame_rate = stream.get("avg_frame_rate") or "0/0"
    fps = 0.0
    if isinstance(avg_frame_rate, str) and "/" in avg_frame_rate:
        num_str, den_str = avg_frame_rate.split("/", 1)
        try:
            num = float(num_str)
            den = float(den_str)
            if den != 0:
                fps = num / den
        except ValueError:
            fps = 0.0
    else:
        try:
            fps = float(avg_frame_rate)
        except (TypeError, ValueError):
            fps = 0.0

    nb_frames_raw = stream.get("nb_frames")
    total_frames = 0
    if isinstance(nb_frames_raw, str) and nb_frames_raw.isdigit():
        total_frames = int(nb_frames_raw)

    return width, height, fps, total_frames


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

        frame = np.frombuffer(buf, dtype=np.uint8)
        try:
            frame = frame.reshape((self.height, self.width, 3))
        except ValueError as exc:
            raise RuntimeError(
                "Decoded raw frame from ffmpeg has unexpected size when reshaping.\n"
                f"Expected {self.height}x{self.width}x3 bytes."
            ) from exc
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
            # Opinionated choice: always encode as MP4 using MPEG-4 Part 2 ('mp4v')
            # rather than guessing based on the input codec.
            "-c:v",
            "mpeg4",
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
                "Ensure 'ffmpeg' is installed and available on PATH."
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
                "explicitly requested 'mpeg4' codec (FourCC 'mp4v'), or the output file "
                "could not be written."
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
            "Offline object tracking with Qwen3-VL-32B-Thinking on ./workspace/input.mp4.\n"
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

    # No other CLI arguments are supported beyond --description and the labeling options, by design.
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

    - Reject extremely skewed aspect ratios (long/short > 200).
    - Reject videos with tiny dimensions (< 10px on any side).
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Video has non-positive dimensions: width={width}, height={height}.")

    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise ValueError(
            f"Video frame dimensions {width}x{height} are too small. "
            f"Qwen3-VL requires width and height to be greater than {MIN_DIMENSION} pixels."
        )

    long_edge = max(width, height)
    short_edge = min(width, height)
    aspect_ratio = long_edge / short_edge

    if aspect_ratio > MAX_ABS_ASPECT_RATIO:
        raise ValueError(
            "No, you can't use that video: its aspect ratio exceeds Qwen3-VL's recommended limit.\n"
            f"  - width x height = {width} x {height}\n"
            f"  - long/short edge ratio = {aspect_ratio:.2f} > {MAX_ABS_ASPECT_RATIO:.0f}\n"
            "The absolute aspect ratio must be <= 200:1 or 1:200 according to the Qwen-VL docs."
        )


def compute_frame_resize(width: int, height: int) -> tuple[int, int]:
    """
    Decide how to resize frames before sending them to the LLM.

    Policy (opinionated):
    - If the long edge is <= 1000 px, keep original resolution.
    - If the long edge is > 1000 px, downscale so that the long edge becomes 1000 px,
      preserving aspect ratio. No cropping is ever performed.

    This keeps us near the 1000x1000 normalized training grid while preserving all content.
    """
    lawyerly_video_checks(width, height)

    long_edge = max(width, height)
    short_edge = min(width, height)

    if long_edge <= 1000:
        # Within the "sweet spot" already; no resize.
        print(
            f"[video] Using original frame resolution {width}x{height} "
            "(long edge <= 1000px, no downscale needed).",
            flush=True,
        )
        return width, height

    scale = 1000.0 / float(long_edge)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    # Safety against rounding to < MIN_DIMENSION on one side.
    if new_width < MIN_DIMENSION or new_height < MIN_DIMENSION:
        raise ValueError(
            "Downscaling the video to keep the long edge at 1000px would result in dimensions "
            f"below {MIN_DIMENSION}px: {new_width}x{new_height}. Refusing to process."
        )

    print(
        "[video] Resizing frames before feeding them to Qwen3-VL:\n"
        f"        original: {width}x{height}\n"
        f"        resized:  {new_width}x{new_height}\n"
        "        rationale: long edge capped at 1000px to align with the model's 1000×1000 "
        "normalized bounding-box grid while preserving aspect ratio.",
        flush=True,
    )
    return new_width, new_height


def frame_to_base64_jpeg(frame_bgr: np.ndarray, target_w: int, target_h: int) -> str:
    """
    Resize frame (if needed) and encode as base64 JPEG for Qwen3-VL.
    """
    h, w = frame_bgr.shape[:2]
    if (w, h) != (target_w, target_h):
        frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # JPEG encode; Qwen3-VL is robust to standard JPEG compression.
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame for LLM input.")
    return base64.b64encode(buf).decode("ascii")


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
    - Each coordinate must be a finite number in [0, 999].
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
            if not (0.0 <= f <= 999.0):
                raise ValueError(
                    f"Entry {idx} coordinate {j} is out of [0, 999] range: {f!r}"
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
) -> None:
    """
    Draw normalized bounding boxes on the frame in-place at the original video resolution.
    """
    height, width = frame_bgr.shape[:2]

    if not bboxes:
        return

    # Dynamic styling for readability.
    min_side = min(width, height)
    thickness = max(2, int(round(min_side / 400)))
    font_scale = max(0.5, min_side / 800.0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box in bboxes:
        # Map from [0, 999] grid to pixel coordinates.
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

        # Box color: red; label background: filled red, text in white.
        color_box = (0, 0, 255)
        color_text = (255, 255, 255)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_box, thickness)

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
        "- Each coordinate is an integer in [0, 999].\n"
        "- x1 < x2 and y1 < y2.\n"
        "- For each detected object, the 'label' field must be EXACTLY one of the valid\n"
        "  labels listed above (match the spelling as closely as possible).\n"
        "- If no matching object exists, return [].\n"
        "- Do NOT output any extra text, comments, or explanations. Only pure JSON."
    )

    user_prompt = (
        "Locate every object that matches the following description in the image and "
        "return their 2D bounding boxes using the required JSON schema.\n\n"
        f"Objects of interest: {description}\n\n"
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
    total_frames: int,
    client,
    connection: OllamaConnectionConfig,
    resized_w: int,
    resized_h: int,
) -> List[NormalizedBBox]:
    """
    Run Qwen3-VL on a single frame, retrying until we get sane bounding boxes.

    The loop has *no* fallback other than asking the LLM again; any malformed
    output leads to another attempt, with the error printed for visibility.
    """
    attempt = 0
    model_name = "qwen3-vl:32b-thinking"

    while True:
        attempt += 1
        b64_image = frame_to_base64_jpeg(frame_bgr, resized_w, resized_h)

        messages = build_llm_messages(b64_image, description, allowed_labels)

        params = ChatCompleteParams(
            messages=messages,
            model=model_name,
            client=client,
            connection=connection,
            max_completion_tokens=512,
            please_no_thinking=False,
            require_json=False,  # thinking models cannot use strict JSON mode
        )

        try:
            response = chat_complete(params)
        except Exception as exc:
            # Non-LLM issues (network, Ollama down) are fatal; we don't silently loop them.
            raise RuntimeError(
                f"Ollama / Qwen3-VL call failed on frame {frame_index + 1}/{total_frames}, "
                f"attempt {attempt}: {exc}"
            ) from exc

        stats = print_stats(response)
        if stats is not None:
            print(
                f"[frame {frame_index + 1}/{total_frames}][attempt {attempt}] "
                f"LLM stats:\n{stats}",
                flush=True,
            )
        else:
            print(
                f"[frame {frame_index + 1}/{total_frames}][attempt {attempt}] "
                "LLM did not return detailed token-level stats.",
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
                f"[frame {frame_index + 1}/{total_frames}][attempt {attempt}] "
                f"LLM output invalid, will retry:\n"
                f"  error: {exc}\n"
                f"  raw (sanitized) preview: {preview}",
                flush=True,
            )
            continue

        print(
            f"[frame {frame_index + 1}/{total_frames}][attempt {attempt}] "
            f"Successfully parsed {len(boxes)} bounding box(es).",
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
            if not (0 <= value <= 999):
                raise ValueError(
                    f"JSONL line {line_number} 'boxes[{idx}].{coord_name}'={value} "
                    "is out of [0, 999] range."
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


def process_video(description: str, allowed_labels: List[str]) -> None:
    if not os.path.exists(INPUT_VIDEO_PATH):
        raise FileNotFoundError(
            f"Expected input video at {INPUT_VIDEO_PATH!r}, but it does not exist."
        )

    orig_width, orig_height, fps, total_frames = probe_video_metadata(INPUT_VIDEO_PATH)

    print(
        f"[video] Loaded {INPUT_VIDEO_PATH}\n"
        f"        resolution: {orig_width}x{orig_height}\n"
        f"        fps:        {fps:.3f}\n"
        f"        frames:     {total_frames}",
        flush=True,
    )

    # Lawyerly checks + resize policy (may raise and abort early).
    resized_w, resized_h = compute_frame_resize(orig_width, orig_height)

    # If base_url is omitted, it defaults to http://127.0.0.1:11434
    connection = OllamaConnectionConfig(base_url="http://172.17.0.1:11434")
    client = get_client(connection)

    try:
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
            os.makedirs(os.path.dirname(FRAMES_JSONL_PATH), exist_ok=True)

        if total_frames > 0 and frames_already_done > total_frames:
            raise RuntimeError(
                "Existing JSONL file appears to contain more frames than the input video reports.\n"
                f"  - video frames (ffprobe nb_frames): {total_frames}\n"
                f"  - JSONL frames:                     {frames_already_done}\n"
                "This script refuses to guess how to reconcile this mismatch. "
                "Delete the JSONL file and re-run, or investigate the inconsistency."
            )

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
                            total_frames=total_frames if total_frames > 0 else frame_index + 1,
                            client=client,
                            connection=connection,
                            resized_w=resized_w,
                            resized_h=resized_h,
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
            decode_stream.close(expect_zero_exit=True)

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

        # Stage 2: now that the JSONL file has a clean entry for every frame, build the video.
        os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

        fps_out = fps if fps > 0 else 25.0

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

                    draw_bboxes_on_frame(frame_bgr, boxes)
                    encode_stream.write_frame(frame_bgr)

                    frame_index += 1

            print(
                f"[done] Processed {frame_index} frame(s). "
                f"Annotated video written to {OUTPUT_VIDEO_PATH!r}.",
                flush=True,
            )
        finally:
            # In the happy path this enforces zero exit codes; in exceptional paths it may
            # raise additional errors which are acceptable since we are already failing loudly.
            decode_stream2.close(expect_zero_exit=True)
            encode_stream.close(expect_zero_exit=True)
    finally:
        client.close()


def main() -> None:
    args = parse_args()
    try:
        process_video(args.description, args.allowed_labels)
    except Exception as exc:
        # Complain loudly and exit non-zero.
        print(f"[fatal] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
