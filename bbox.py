from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
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


def process_video(description: str, allowed_labels: List[str]) -> None:
    if not os.path.exists(INPUT_VIDEO_PATH):
        raise FileNotFoundError(
            f"Expected input video at {INPUT_VIDEO_PATH!r}, but it does not exist."
        )

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {INPUT_VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"[video] Loaded {INPUT_VIDEO_PATH}\n"
        f"        resolution: {orig_width}x{orig_height}\n"
        f"        fps:        {fps:.3f}\n"
        f"        frames:     {total_frames}",
        flush=True,
    )

    # Lawyerly checks + resize policy (may raise and abort early).
    resized_w, resized_h = compute_frame_resize(orig_width, orig_height)
    # Prepare output writer with original resolution & fps to preserve quality.
    # Opinionated choice: always encode as MP4 using MPEG-4 Part 2 ('mp4v'),
    # instead of guessing based on the input codec. If your OpenCV/FFmpeg build
    # cannot encode 'mp4v', we fail loudly instead of silently falling back.
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_out = fps if fps > 0 else 25.0

    out = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        fourcc,
        fps_out,
        (orig_width, orig_height),
    )
    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            "Failed to open output video for writing.\n"
            "Reason: your OpenCV/FFmpeg build does not provide an encoder for the "
            "explicitly requested 'mp4v' codec.\n"
            "This script refuses to guess or fall back to other codecs. "
            "Install/rebuild FFmpeg/OpenCV with an MP4 encoder (e.g. libx264/mp4v) "
            "or change the hard-coded codec in bbox.py."
        )

    # If base_url is omitted, it defaults to http://127.0.0.1:11434
    connection = OllamaConnectionConfig(base_url="http://172.17.0.1:11434")
    client = get_client(connection)

    try:
        frame_index = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break  # End of video; do not rely solely on frame_count.

            # Safety check: if frame resolution ever differs, we bail out loudly.
            h, w = frame_bgr.shape[:2]
            if (w, h) != (orig_width, orig_height):
                raise RuntimeError(
                    "Encountered a frame with different resolution within the same video. "
                    f"Expected {orig_width}x{orig_height}, got {w}x{h} at frame {frame_index}."
                )

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

            draw_bboxes_on_frame(frame_bgr, boxes)
            out.write(frame_bgr)

            frame_index += 1

        print(
            f"[done] Processed {frame_index} frame(s). "
            f"Annotated video written to {OUTPUT_VIDEO_PATH!r}.",
            flush=True,
        )
    finally:
        cap.release()
        out.release()
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
