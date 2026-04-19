"""
VLM Orchestrator - Pipeline implementation.

Architecture:
  Audio path  : on_audio -> background transcription via Gemini ->
                TranscriptBuffer + correction-keyword scan -> error_detected.
  Frame path  : on_frame -> motion+SSIM heuristic gate ->
                Tier 1 VLM (with transcript context) -> optional Tier 2 ->
                state machine -> step_completion. SSIM-stable streak ->
                idle_detected.
  Threading   : ThreadPoolExecutor runs transcription and Tier 2 escalations
                so on_frame / on_audio return fast.
"""

import json
import os
import sys
import re
import io
import time
import random
import wave
import base64
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import requests
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
_loaded = load_dotenv(dotenv_path=_env_path)
if not _loaded:
    # Fallback: try CWD-based search
    _loaded = load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import StreamingHarness
from src.data_loader import load_procedure_json, validate_procedure_format


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

TIER1_MODEL = "google/gemini-2.5-flash-lite"
TIER2_MODEL = "google/gemini-2.5-flash"
AUDIO_MODEL = "google/gemini-2.5-flash"

FALLBACK_MODEL = "google/gemini-2.5-flash"

RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_RETRIES = 3
BASE_BACKOFF_SEC = 1.0

DEFAULT_TIMEOUT_SEC = 30
AUDIO_TIMEOUT_SEC = 45

CORRECTION_PATTERN = re.compile(
    r"\b(no|stop|wait|wrong|don't|do not|hold on|careful|not like that|"
    r"that's not|incorrect|mistake|nope|hang on|undo)\b",
    re.IGNORECASE,
)


# ==========================================================================
# VLM API HELPERS
# ==========================================================================

def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator Evaluation",
    }


def _extract_content(resp_json: Dict[str, Any]) -> str:
    try:
        return resp_json["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _post_with_retry(
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: int,
    label: str,
) -> Optional[requests.Response]:
    """POST with exponential backoff on transient errors. Returns final response or None."""
    last_err: Optional[str] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_err = f"network: {exc}"
            if attempt >= MAX_RETRIES:
                print(f"  [{label}] giving up after {attempt + 1} attempts: {last_err}")
                return None
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code not in RETRY_STATUSES or attempt >= MAX_RETRIES:
                body = (resp.text or "")[:400]
                print(f"  [{label}] HTTP {resp.status_code}: {body}")
                return resp
            last_err = f"HTTP {resp.status_code}"

        sleep_s = BASE_BACKOFF_SEC * (2 ** attempt) + random.uniform(0, 0.5)
        time.sleep(sleep_s)

    return None


def _post_json(
    payload: Dict[str, Any],
    api_key: str,
    timeout: int,
    label: str,
) -> Optional[Dict[str, Any]]:
    resp = _post_with_retry(payload, _build_headers(api_key), timeout, label)
    if resp is None or resp.status_code >= 400:
        return None
    try:
        return resp.json()
    except ValueError as exc:
        print(f"  [{label}] bad JSON body: {exc}")
        return None


def call_vlm(
    api_key: str,
    frame_base64: str,
    prompt: str,
    model: str = "google/gemini-2.5-flash",
    stream: bool = False,
) -> str:
    """Call a vision VLM via OpenRouter with a single frame.

    Retries on 429/5xx. On 404 (model not found), falls back to FALLBACK_MODEL.
    Streaming path is kept for compatibility but is NOT retried.
    """
    headers = _build_headers(api_key)
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
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, stream=True, timeout=DEFAULT_TIMEOUT_SEC)
        if resp.status_code >= 400:
            print(f"  [vlm-stream] HTTP {resp.status_code}: {(resp.text or '')[:300]}")
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

    data = _post_json(payload, api_key, DEFAULT_TIMEOUT_SEC, f"vlm:{model}")
    if data is None and model != FALLBACK_MODEL:
        payload["model"] = FALLBACK_MODEL
        print(f"  [vlm] falling back to {FALLBACK_MODEL}")
        data = _post_json(payload, api_key, DEFAULT_TIMEOUT_SEC, f"vlm:{FALLBACK_MODEL}")
    if data is None:
        return ""
    return _extract_content(data)


def call_vlm_audio(
    api_key: str,
    audio_base64_wav: str,
    prompt: str,
    model: str = AUDIO_MODEL,
) -> str:
    """Call a multimodal VLM with a WAV audio chunk (base64-encoded)."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_base64_wav, "format": "wav"},
                    },
                ],
            }
        ],
    }
    data = _post_json(payload, api_key, AUDIO_TIMEOUT_SEC, f"audio:{model}")
    if data is None and model != FALLBACK_MODEL:
        payload["model"] = FALLBACK_MODEL
        print(f"  [audio] falling back to {FALLBACK_MODEL}")
        data = _post_json(payload, api_key, AUDIO_TIMEOUT_SEC, f"audio:{FALLBACK_MODEL}")
    if data is None:
        return ""
    return _extract_content(data)


def probe_openrouter(api_key: str) -> bool:
    """Single cheap text call to verify key + connectivity. True on success."""
    payload = {
        "model": TIER1_MODEL,
        "messages": [{"role": "user", "content": "Reply with: ok"}],
        "max_tokens": 8,
    }
    data = _post_json(payload, api_key, 15, "probe")
    if data is None:
        return False
    content = _extract_content(data).strip().lower()
    return bool(content)


def pcm16_to_wav_base64(pcm: bytes, sample_rate: int = 16000) -> str:
    """Wrap raw 16-bit mono PCM as a WAV container, return base64 string."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_json_block(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ==========================================================================
# TRANSCRIPT BUFFER (audio context shared across threads)
# ==========================================================================

class TranscriptBuffer:
    """Thread-safe rolling transcript with timestamped segments."""

    def __init__(self) -> None:
        self._segments: List[Tuple[float, float, str, bool]] = []
        self._lock = threading.Lock()

    def append(self, start_sec: float, end_sec: float, text: str, has_correction: bool) -> None:
        with self._lock:
            self._segments.append((start_sec, end_sec, text, has_correction))

    def recent_text(self, since_sec: float, until_sec: float) -> str:
        with self._lock:
            parts = [t for (s, e, t, _) in self._segments if e >= since_sec and s <= until_sec and t]
        return " ".join(parts).strip()

    def latest_correction(self) -> Optional[Tuple[float, float, str]]:
        with self._lock:
            for (s, e, t, c) in reversed(self._segments):
                if c:
                    return (s, e, t)
        return None


# ==========================================================================
# PIPELINE
# ==========================================================================

class Pipeline:
    """VLM orchestration pipeline.

    Heuristic-gated cascade: cheap motion/SSIM filter -> Tier 1 VLM with
    transcript context -> Tier 2 escalation when uncertain. Audio chunks
    feed a transcript buffer and trigger error_detected on instructor
    corrections.
    """

    HEURISTIC_DIFF_THRESHOLD = 8.0
    HEURISTIC_SSIM_THRESHOLD = 0.88
    IDLE_SSIM_THRESHOLD = 0.95
    IDLE_MIN_DURATION = 4.0

    TIER1_MIN_INTERVAL = 1.5
    TIER2_MIN_INTERVAL = 4.0

    TIER1_LOW_CONFIDENCE = 0.6
    TRANSCRIPT_CONTEXT_WINDOW = 12.0
    
    MAX_VLM_CALLS = 200

    def __init__(self, harness: StreamingHarness, api_key: str, procedure: Dict[str, Any]):
        self.harness = harness
        self.api_key = api_key
        self.procedure = procedure
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps: List[Dict[str, Any]] = procedure["steps"]

        # Heuristic state
        self.prev_gray: Optional[np.ndarray] = None
        self.last_diff: float = 0.0
        self.last_ssim: float = 1.0

        # State machine
        self.current_step_idx: int = 0
        self.completed_steps: set = set()
        self.last_emitted_step_at: Dict[int, float] = {}
        self._state_lock = threading.Lock()

        # Idle tracking
        self._idle_start: Optional[float] = None
        self._idle_emitted: bool = False

        # VLM rate limiting
        self._last_tier1_t: float = -1e9
        self._last_tier2_t: float = -1e9
        self.total_vlm_calls: int = 0

        # Audio + transcript
        self.transcript = TranscriptBuffer()
        self._executor = ThreadPoolExecutor(max_workers=3)
        self._last_frame_b64: Optional[str] = None
        self._last_frame_t: float = 0.0
        self._error_emit_lock = threading.Lock()
        self._last_error_emit_t: float = -1e9

        # Analysis trace
        self.analysis_path = Path(__file__).parent.parent / "data" / "analysis" / "diff_ssim_dslr2.csv"
        self.analysis_path.parent.mkdir(parents=True, exist_ok=True)
        self._analysis_rows: List[dict] = []

    # ----------------------------------------------------------------------
    # Frame callback
    # ----------------------------------------------------------------------

    def on_frame(self, frame: np.ndarray, timestamp_sec: float, frame_base64: str) -> None:
        small = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        diff = 0.0
        score = 1.0
        if self.prev_gray is not None:
            diff = float(np.mean(cv2.absdiff(gray, self.prev_gray)))
            score, _ = ssim(self.prev_gray, gray, full=True)

        self.prev_gray = gray
        self.last_diff = diff
        self.last_ssim = score
        self._last_frame_b64 = frame_base64
        self._last_frame_t = timestamp_sec
        self._analysis_rows.append({"timestamp_sec": timestamp_sec, "diff": diff, "ssim": score})

        with self._state_lock:
            done = self.current_step_idx >= len(self.steps)
        if done:
            return

        # Idle detection (heuristic only; cheap)
        self._update_idle(timestamp_sec, score, diff)

        # Heuristic gate for VLM
        active = diff > self.HEURISTIC_DIFF_THRESHOLD or score < self.HEURISTIC_SSIM_THRESHOLD
        if not active:
            return

        if (timestamp_sec - self._last_tier1_t) < self.TIER1_MIN_INTERVAL:
            return
        if self.total_vlm_calls >= self.MAX_VLM_CALLS:
            return

        self._last_tier1_t = timestamp_sec
        self._executor.submit(self._run_tier1, frame_base64, timestamp_sec)

    # ----------------------------------------------------------------------
    # Audio callback
    # ----------------------------------------------------------------------

    def on_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float) -> None:
        if not audio_bytes:
            return
        self._executor.submit(self._transcribe_chunk, audio_bytes, start_sec, end_sec)

    # ----------------------------------------------------------------------
    # Tier 1 VLM
    # ----------------------------------------------------------------------

    def _run_tier1(self, frame_b64: str, timestamp_sec: float) -> None:
        prompt = self._build_step_prompt(timestamp_sec, tier=1)
        try:
            self.total_vlm_calls += 1
            raw = call_vlm(self.api_key, frame_b64, prompt, model=TIER1_MODEL)
        except Exception as exc:
            print(f"  [tier1] error at {timestamp_sec:.1f}s: {exc}")
            return

        parsed = parse_json_block(raw) or {}
        decision = parsed.get("decision", "no_change")
        step_id = parsed.get("step_id")
        try:
            confidence = float(parsed.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        observation = (parsed.get("observation") or "")[:200]

        ambiguous = decision == "ambiguous" or (
            decision == "step_complete" and confidence < self.TIER1_LOW_CONFIDENCE
        )

        if ambiguous and (timestamp_sec - self._last_tier2_t) >= self.TIER2_MIN_INTERVAL:
            self._last_tier2_t = timestamp_sec
            self._executor.submit(self._run_tier2, frame_b64, timestamp_sec, observation)
            return

        if decision == "step_complete" and isinstance(step_id, int):
            self._emit_step_completion(step_id, timestamp_sec, confidence, observation, source="video")

    # ----------------------------------------------------------------------
    # Tier 2 VLM (background)
    # ----------------------------------------------------------------------

    def _run_tier2(self, frame_b64: str, timestamp_sec: float, tier1_obs: str) -> None:
        prompt = self._build_step_prompt(timestamp_sec, tier=2, tier1_obs=tier1_obs)
        try:
            self.total_vlm_calls += 1
            raw = call_vlm(self.api_key, frame_b64, prompt, model=TIER2_MODEL)
        except Exception as exc:
            print(f"  [tier2] error at {timestamp_sec:.1f}s: {exc}")
            return

        parsed = parse_json_block(raw) or {}
        decision = parsed.get("decision", "no_change")
        step_id = parsed.get("step_id")
        try:
            confidence = float(parsed.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        observation = (parsed.get("observation") or "")[:200]

        if decision == "step_complete" and isinstance(step_id, int):
            self._emit_step_completion(step_id, timestamp_sec, confidence, observation, source="both")

    # ----------------------------------------------------------------------
    # Audio transcription (background)
    # ----------------------------------------------------------------------

    def _transcribe_chunk(self, pcm: bytes, start_sec: float, end_sec: float) -> None:
        try:
            wav_b64 = pcm16_to_wav_base64(pcm)
        except Exception as exc:
            print(f"  [audio] wrap error {start_sec:.1f}s: {exc}")
            return

        prompt = (
            "Transcribe this short instructor/student speech verbatim. "
            "Reply ONLY as JSON: "
            '{"transcript": "<verbatim text or empty>", '
            '"contains_correction": <true|false>, '
            '"reason": "<short reason if correction>"} '
            "A correction is the instructor telling the student to stop, undo, or "
            "redo something (\"no\", \"stop\", \"wait\", \"that's wrong\", \"undo\"). "
            "If the audio has no speech, return empty transcript."
        )
        try:
            self.total_vlm_calls += 1
            raw = call_vlm_audio(self.api_key, wav_b64, prompt, model=AUDIO_MODEL)
        except Exception as exc:
            print(f"  [audio] vlm error {start_sec:.1f}s: {exc}")
            return

        parsed = parse_json_block(raw) or {}
        text = (parsed.get("transcript") or "").strip()
        flagged = bool(parsed.get("contains_correction"))
        reason = (parsed.get("reason") or "").strip()

        # Backstop: regex scan in case the model misses
        keyword_hit = bool(CORRECTION_PATTERN.search(text)) if text else False
        has_correction = flagged or keyword_hit

        self.transcript.append(start_sec, end_sec, text, has_correction)

        if has_correction:
            self._handle_audio_correction(start_sec, end_sec, text, reason)

    # ----------------------------------------------------------------------
    # Error handling from audio
    # ----------------------------------------------------------------------

    def _handle_audio_correction(self, start_sec: float, end_sec: float, text: str, reason: str) -> None:
        with self._error_emit_lock:
            if (start_sec - self._last_error_emit_t) < 3.0:
                return
            self._last_error_emit_t = start_sec

        # The mistake itself happened ~2-5s before the correction was spoken.
        error_ts = max(0.0, start_sec - 3.0)

        # Single-emit policy: classify with frame if available, otherwise emit
        # audio-only. Exactly one error_detected event per correction.
        if self._last_frame_b64 is not None:
            frame_b64 = self._last_frame_b64
            self._executor.submit(
                self._classify_error_with_frame,
                frame_b64,
                error_ts,
                text,
                reason,
            )
        else:
            self._emit_audio_only_error(error_ts, text, reason)

    def _emit_audio_only_error(self, error_ts: float, text: str, reason: str) -> None:
        spoken = self._build_spoken_response(text, reason)
        try:
            self.harness.emit_event({
                "timestamp_sec": error_ts,
                "type": "error_detected",
                "error_type": "wrong_action",
                "severity": "warning",
                "confidence": 0.7,
                "source": "audio",
                "description": f"Instructor correction: {text[:120]}",
                "vlm_observation": reason or text[:160],
                "spoken_response": spoken,
            })
        except ValueError as exc:
            print(f"  [error] emit rejected: {exc}")

    def _classify_error_with_frame(
        self,
        frame_b64: str,
        error_ts: float,
        transcript_snippet: str,
        reason: str = "",
    ) -> None:
        steps_block = self._format_step_window()
        prompt = (
            f"Task: {self.task_name}\n"
            f"Expected upcoming steps:\n{steps_block}\n\n"
            f"Instructor just said: \"{transcript_snippet}\".\n"
            "Look at this frame from when the mistake likely occurred. "
            "Reply ONLY as JSON: "
            '{"error_type": "wrong_action|wrong_sequence|safety_violation|improper_technique|other", '
            '"severity": "info|warning|critical", '
            '"description": "<one short sentence>", '
            '"spoken_response": "<short coaching sentence to the student>"}'
        )
        raw = ""
        try:
            self.total_vlm_calls += 1
            raw = call_vlm(self.api_key, frame_b64, prompt, model=TIER2_MODEL)
        except Exception as exc:
            print(f"  [error-classify] {exc}")

        parsed = parse_json_block(raw) if raw else None
        if not parsed:
            # Classification failed or returned nothing: fall back to audio-only
            # emit so the event is still recorded exactly once.
            self._emit_audio_only_error(error_ts, transcript_snippet, reason)
            return

        try:
            self.harness.emit_event({
                "timestamp_sec": error_ts,
                "type": "error_detected",
                "error_type": parsed.get("error_type") or "wrong_action",
                "severity": parsed.get("severity") or "warning",
                "confidence": 0.8,
                "source": "both",
                "description": (parsed.get("description") or transcript_snippet)[:200],
                "vlm_observation": (parsed.get("description") or "")[:200],
                "spoken_response": parsed.get("spoken_response") or self._build_spoken_response(transcript_snippet, reason),
            })
        except ValueError as exc:
            print(f"  [error-classify] emit rejected: {exc}")

    # ----------------------------------------------------------------------
    # Idle detection (heuristic-only)
    # ----------------------------------------------------------------------

    def _update_idle(self, timestamp_sec: float, score: float, diff: float) -> None:
        if score >= self.IDLE_SSIM_THRESHOLD and diff < 3.0:
            if self._idle_start is None:
                self._idle_start = timestamp_sec
                self._idle_emitted = False
            elif (
                not self._idle_emitted
                and (timestamp_sec - self._idle_start) >= self.IDLE_MIN_DURATION
            ):
                self._idle_emitted = True
                try:
                    self.harness.emit_event({
                        "timestamp_sec": timestamp_sec,
                        "type": "idle_detected",
                        "confidence": 0.6,
                        "source": "video",
                        "description": f"No motion for ~{timestamp_sec - self._idle_start:.1f}s",
                    })
                except ValueError as exc:
                    print(f"  [idle] emit rejected: {exc}")
        else:
            self._idle_start = None
            self._idle_emitted = False

    # ----------------------------------------------------------------------
    # State machine
    # ----------------------------------------------------------------------

    def _emit_step_completion(
        self,
        step_id: int,
        timestamp_sec: float,
        confidence: float,
        observation: str,
        source: str,
    ) -> None:
        with self._state_lock:
            if step_id in self.completed_steps:
                return
            if not (1 <= step_id <= len(self.steps)):
                return

            expected_idx = self.current_step_idx
            target_idx = step_id - 1
            # Only accept current step or one ahead.
            if target_idx < expected_idx or target_idx > expected_idx + 1:
                return

            # Mark intermediate skipped step as completed implicitly.
            for skipped_idx in range(expected_idx, target_idx):
                sid = self.steps[skipped_idx]["step_id"]
                self.completed_steps.add(sid)

            description = self.steps[target_idx].get("description", "")
            self.completed_steps.add(step_id)
            self.last_emitted_step_at[step_id] = timestamp_sec
            self.current_step_idx = target_idx + 1
            self._idle_start = None
            self._idle_emitted = False

        try:
            self.harness.emit_event({
                "timestamp_sec": timestamp_sec,
                "type": "step_completion",
                "step_id": step_id,
                "confidence": max(0.0, min(1.0, confidence or 0.5)),
                "source": source,
                "description": description,
                "vlm_observation": observation,
            })
        except ValueError as exc:
            print(f"  [step] emit rejected: {exc}")

    # ----------------------------------------------------------------------
    # Prompt building
    # ----------------------------------------------------------------------

    def _format_step_window(self) -> str:
        with self._state_lock:
            idx = self.current_step_idx
        window = self.steps[idx : idx + 3]
        if not window:
            return "(all steps already completed)"
        lines = []
        for step in window:
            lines.append(f"  - step_id {step['step_id']}: {step['description']}")
        return "\n".join(lines)

    def _build_step_prompt(self, timestamp_sec: float, tier: int, tier1_obs: str = "") -> str:
        since = max(0.0, timestamp_sec - self.TRANSCRIPT_CONTEXT_WINDOW)
        transcript_ctx = self.transcript.recent_text(since, timestamp_sec) or "(no recent speech)"

        with self._state_lock:
            completed = sorted(self.completed_steps)
        completed_repr = completed if completed else "(none)"

        steps_block = self._format_step_window()
        last_correction = self.transcript.latest_correction()
        correction_line = ""
        if last_correction:
            cs, _ce, ct = last_correction
            if cs >= timestamp_sec - 15.0:
                correction_line = f"Recent instructor correction at {cs:.1f}s: \"{ct[:120]}\".\n"

        tier_hint = ""
        if tier == 2:
            tier_hint = (
                "A cheaper model was uncertain. Examine the frame carefully and only "
                "claim a step is complete if you can clearly justify it.\n"
                f"Tier-1 noted: {tier1_obs}\n"
            )

        return (
            f"You are watching a technician perform task: {self.task_name}.\n"
            f"Already completed step_ids: {completed_repr}.\n"
            f"Candidate next steps (only choose one of these):\n{steps_block}\n\n"
            f"Recent speech transcript (last ~{int(self.TRANSCRIPT_CONTEXT_WINDOW)}s): \"{transcript_ctx}\"\n"
            f"{correction_line}"
            f"{tier_hint}"
            "Decide whether the LATEST visible action just COMPLETED one of the candidate "
            "next steps in this frame. Be conservative: only mark complete if the action "
            "looks finished (not in progress).\n\n"
            "Reply ONLY as JSON: "
            '{"decision": "step_complete|in_progress|no_change|ambiguous", '
            '"step_id": <int or null>, '
            '"confidence": <0..1>, '
            '"observation": "<one short sentence>"}'
        )

    def _build_spoken_response(self, transcript_text: str, reason: str) -> str:
        if reason:
            return f"Heads up - {reason}. Pause and double-check before continuing."
        return "Hold on - re-check the last step before moving on."

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        if self._analysis_rows:
            try:
                pd.DataFrame(self._analysis_rows).to_csv(self.analysis_path, index=False)
            except Exception as exc:
                print(f"  [analysis] csv write failed: {exc}")


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

    print("  Probing OpenRouter...")
    if not probe_openrouter(api_key):
        print("  ERROR: OpenRouter probe failed. Check API key and network.")
        sys.exit(2)
    print("  OpenRouter OK")

    harness = StreamingHarness(
        video_path=args.video,
        procedure_path=args.procedure,
        speed=args.speed,
        frame_fps=args.frame_fps,
        audio_chunk_sec=args.audio_chunk_sec,
    )

    pipeline = Pipeline(harness, api_key, procedure)

    harness.on_frame(pipeline.on_frame)
    harness.on_audio(pipeline.on_audio)

    try:
        results = harness.run()
    finally:
        pipeline.shutdown()

    harness.save_results(results, args.output)

    print()
    print(f"  Output: {args.output}")
    print(f"  Events: {len(results.events)}")
    print()
    print(f"  Total VLM calls: {pipeline.total_vlm_calls}")
    if not results.events:
        print("  WARNING: No events detected.")


if __name__ == "__main__":
    main()
