"""
VLM Orchestrator — Starter Template

This is where you implement your pipeline. The harness feeds you frames
and audio in real-time. You call VLMs, detect events, and emit them back.

Usage:
    python src/run.py \\
        --procedure data/clip_procedures/CLIP.json \\
        --video path/to/Video_pitchshift.mp4 \\
        --output output/events.json \\
        --speed 1.0
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from dataclasses import asdict

import requests
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import StreamingHarness
from src.data_loader import load_procedure_json, validate_procedure_format


# ==========================================================================
# VLM API HELPER (provided — feel free to modify)
# ==========================================================================

def call_vlm(
    api_key: str,
    frame_base64: str,
    prompt: str,
    model: str = "google/gemini-2.5-flash",
    stream: bool = False,
) -> str:
    """
    Call a VLM via OpenRouter.

    Args:
        api_key: OpenRouter API key
        frame_base64: Base64-encoded JPEG frame
        prompt: Text prompt
        model: OpenRouter model string
        stream: If True, use streaming (SSE) responses for lower time-to-first-token

    Returns:
        Model response text
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator Evaluation",
    }
    payload = {
        "model": model,
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
                    },
                ],
            }
        ],
    }

    if stream:
        # Streaming: read SSE chunks as they arrive
        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
        resp.raise_for_status()
        full_text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        full_text += delta["content"]
                except (json.JSONDecodeError, KeyError):
                    pass
        return full_text
    else:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ==========================================================================
# YOUR PIPELINE — IMPLEMENT THESE CALLBACKS
# ==========================================================================

class Pipeline:
    """
    Your VLM orchestration pipeline.

    The harness calls on_frame() and on_audio() in real-time as the video plays.
    When you detect an event, call self.harness.emit_event({...}).

    Key design decisions you need to make:
    - Which frames to send to the VLM (not every frame — budget is limited)
    - Whether/how to use audio (speech-to-text for instructor corrections?)
    - Which model to use and when (cheap for easy frames, expensive for hard ones?)
    - How to track procedure state (current step, completed steps)
    - How to generate spoken responses for errors
    """

    def __init__(self, harness: StreamingHarness, api_key: str, procedure: Dict[str, Any]):
        self.harness = harness
        self.api_key = api_key
        self.procedure = procedure
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps = procedure["steps"]

        # TODO: Initialize your pipeline state here
        # Examples:
        #   self.current_step = 0
        #   self.completed_steps = set()
        #   self.frame_buffer = []
        #   self.last_activity_time = 0
        #   self.api_calls = 0
        #   self.total_cost = 0

    def on_frame(self, frame: np.ndarray, timestamp_sec: float, frame_base64: str):
        """
        Called by the harness for each video frame.

        Args:
            frame: BGR numpy array (raw frame)
            timestamp_sec: Current video timestamp
            frame_base64: Pre-encoded JPEG base64 string (ready for VLM API)

        TODO: Implement your frame processing logic.
        When you detect an event, call:
            self.harness.emit_event({
                "timestamp_sec": timestamp_sec,
                "type": "step_completion",  # or "error_detected" or "idle_detected"
                "step_id": 1,
                "confidence": 0.9,
                "description": "...",
                "source": "video",
                "vlm_observation": "...",
                # For errors, also include:
                "spoken_response": "Stop — you need to turn off the power first.",
            })
        """
        pass  # TODO: Implement

    def on_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float):
        """
        Called by the harness for each audio chunk.

        Args:
            audio_bytes: Raw PCM audio (16kHz, mono, 16-bit)
            start_sec: Chunk start time in video
            end_sec: Chunk end time in video

        TODO: Implement your audio processing logic.
        Consider: speech-to-text, keyword detection, silence detection.
        The instructor's verbal corrections are a strong signal for errors.
        """
        pass  # TODO: Implement


# ==========================================================================
# MAIN ENTRY POINT
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="VLM Orchestrator Pipeline")
    parser.add_argument("--procedure", required=True, help="Path to procedure JSON")
    parser.add_argument("--video", required=True, help="Path to video MP4 (with audio)")
    parser.add_argument("--output", default="output/events.json", help="Output JSON path")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (1.0 = real-time, 2.0 = 2x, etc.)")
    parser.add_argument("--frame-fps", type=float, default=2.0,
                        help="Frames per second delivered to pipeline (default: 2)")
    parser.add_argument("--audio-chunk-sec", type=float, default=5.0,
                        help="Audio chunk duration in seconds (default: 5)")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs only")
    args = parser.parse_args()

    # Load procedure
    print("=" * 60)
    print("  VLM ORCHESTRATOR")
    print("=" * 60)
    print()

    procedure = load_procedure_json(args.procedure)
    validate_procedure_format(procedure)
    task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
    print(f"  Procedure: {task_name} ({len(procedure['steps'])} steps)")
    print(f"  Video:     {args.video}")
    print(f"  Speed:     {args.speed}x")
    print()

    if args.dry_run:
        if not Path(args.video).exists():
            print(f"  WARNING: Video not found: {args.video}")
            print("  [DRY RUN] Procedure validated. Video not checked (file missing).")
        else:
            print("  [DRY RUN] Inputs validated. Skipping pipeline.")
        return

    if not Path(args.video).exists():
        print(f"  ERROR: Video not found: {args.video}")
        sys.exit(1)

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("  ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        sys.exit(1)

    # Create harness and pipeline
    harness = StreamingHarness(
        video_path=args.video,
        procedure_path=args.procedure,
        speed=args.speed,
        frame_fps=args.frame_fps,
        audio_chunk_sec=args.audio_chunk_sec,
    )

    pipeline = Pipeline(harness, api_key, procedure)

    # Register callbacks
    harness.on_frame(pipeline.on_frame)
    harness.on_audio(pipeline.on_audio)

    # Run
    results = harness.run()

    # Save
    harness.save_results(results, args.output)

    print()
    print(f"  Output: {args.output}")
    print(f"  Events: {len(results.events)}")
    print()

    if not results.events:
        print("  WARNING: No events detected. Implement Pipeline.on_frame() and Pipeline.on_audio().")


if __name__ == "__main__":
    main()
