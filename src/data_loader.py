"""
VLM Orchestrator - Data Loading Module
Handles video streaming from URLs or local files and procedure JSON loading.
"""

import json
import cv2
import numpy as np
import requests
from pathlib import Path
from typing import Generator, Dict, Any, Optional, Tuple
import io
import tempfile
from PIL import Image


class VideoStream:
    """
    A video stream loader that supports both local files and remote URLs.
    Yields frames at a configurable frame rate.
    """

    def __init__(
        self,
        source: str,
        target_fps: int = 1,
        max_frames: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize video stream.

        Args:
            source: Path to local video file or URL
            target_fps: Frame rate to sample at (default: 1 FPS)
            max_frames: Maximum frames to yield (None = all frames)
            verbose: Print debug information
        """
        self.source = source
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.verbose = verbose
        self._is_url = source.startswith("http://") or source.startswith("https://")
        self._frame_count = 0
        self._current_time_sec = 0.0

    def _get_local_video_properties(self) -> Tuple[int, float, int, int]:
        """
        Get properties from local video file.
        Returns: (total_frames, fps, width, height)
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.source}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()
        return total_frames, fps, width, height

    def get_properties(self) -> Dict[str, Any]:
        """
        Get video properties (fps, resolution, duration).
        """
        if self._is_url:
            # For remote videos, we may not know exact properties until streaming
            # Return best-effort estimates
            return {
                "source": self.source,
                "target_fps": self.target_fps,
                "is_remote": True,
                "note": "Properties are approximate for remote sources",
            }
        else:
            try:
                total_frames, fps, width, height = self._get_local_video_properties()
                duration_sec = total_frames / fps if fps > 0 else 0
                return {
                    "source": self.source,
                    "total_frames": total_frames,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "duration_sec": duration_sec,
                    "is_remote": False,
                }
            except Exception as e:
                raise ValueError(f"Error reading video properties: {e}")

    def _stream_from_url(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Stream frames from a remote URL.
        Yields: (frame as numpy array, timestamp in seconds)
        """
        try:
            response = requests.get(self.source, stream=True, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch video from URL: {e}")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(response.content)
            temp_path = tmp.name
        try:
            yield from self._stream_from_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _stream_from_file(self, filepath: str) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Stream frames from a local file.
        Yields: (frame as numpy array, timestamp in seconds)
        """
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {filepath}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default fallback

        frame_interval = int(fps / self.target_fps) if self.target_fps > 0 else 1
        frame_count = 0
        yielded_count = 0

        if self.verbose:
            print(f"Streaming from {filepath} at {self.target_fps} FPS (interval: {frame_interval})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Yield frames at target FPS interval
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                self._current_time_sec = timestamp

                if self.max_frames and yielded_count >= self.max_frames:
                    break

                yield frame, timestamp
                yielded_count += 1

            frame_count += 1

        cap.release()

        if self.verbose:
            print(f"Finished streaming: {yielded_count} frames yielded")

    def stream_frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Main generator that yields frames and timestamps.

        Yields:
            (frame, timestamp_sec): BGR frame as numpy array and timestamp in seconds
        """
        if self._is_url:
            yield from self._stream_from_url()
        else:
            yield from self._stream_from_file(self.source)


def load_procedure_json(filepath: str) -> Dict[str, Any]:
    """
    Load procedure/SOP JSON file.

    Args:
        filepath: Path to the procedure JSON file

    Returns:
        Dictionary containing the procedure specification
    """
    try:
        with open(filepath, "r") as f:
            procedure = json.load(f)
        return procedure
    except FileNotFoundError:
        raise FileNotFoundError(f"Procedure file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in procedure file: {e}")


def validate_procedure_format(procedure: Dict[str, Any]) -> bool:
    """
    Validate that a procedure dictionary has the expected format.

    Args:
        procedure: Procedure dictionary to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Accept either "task" or "task_name" for the task label
    if "task" not in procedure and "task_name" not in procedure:
        raise ValueError("Procedure missing required field: 'task' or 'task_name'")
    if "steps" not in procedure:
        raise ValueError("Procedure missing required field: 'steps'")

    if not isinstance(procedure["steps"], list):
        raise ValueError("Procedure 'steps' must be a list")

    for i, step in enumerate(procedure["steps"]):
        required_step_fields = ["step_id", "description"]
        for field in required_step_fields:
            if field not in step:
                raise ValueError(f"Step {i} missing required field: {field}")

    return True


def frame_to_base64(frame: np.ndarray) -> str:
    """
    Convert an OpenCV BGR frame to base64-encoded JPEG for VLM API calls.

    Args:
        frame: BGR frame as numpy array

    Returns:
        Base64-encoded string
    """
    import base64

    # Convert BGR to RGB for Pillow
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)

    # Encode to JPEG
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    # Encode to base64
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str


if __name__ == "__main__":
    # Example usage
    print("VLM Orchestrator - Data Loader Module")
    print("Use: from src.data_loader import VideoStream, load_procedure_json")
