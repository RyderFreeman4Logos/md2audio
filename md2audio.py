"""Convert markdown files to audio (and optionally subtitles/video) via edge-tts."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import edge_tts
from md2audio_dialogue import (
    DialogueTurn,
    fetch_voices,
    load_speaker_config,
    parse_dialogue,
    resolve_auto_voice_map,
    resolve_config_voice_map,
)

DEFAULT_VOICE = "zh-CN-YunyangNeural"
DEFAULT_SPEED = 1.0
_SRT_MAX_CHARS = 35
_SRT_MIN_CHARS = 10
_SRT_SPLIT_TARGET = 20
_TICKS_PER_SECOND = 10_000_000
_DIALOGUE_CACHE_SCHEMA_VERSION = 2
_DIALOGUE_RETRY_ATTEMPTS = 5
_DIALOGUE_MAX_RETRY_DELAY_SECONDS = 8
_DIALOGUE_TURN_TIMEOUT_SECONDS = 60
_AIOHTTP_RETRYABLE_ERROR_NAMES = frozenset({
    "ClientConnectionError",
    "ClientOSError",
    "ClientPayloadError",
    "ServerDisconnectedError",
    "ServerTimeoutError",
})


# ---------------------------------------------------------------------------
# Markdown stripping
# ---------------------------------------------------------------------------

def strip_markdown(text: str) -> str:
    """Remove markdown syntax, keeping readable plain text."""
    text = re.split(
        r"^#{1,6}\s*\**(?:引用的著作|参考文献|References?|Bibliography)\**\s*$",
        text, maxsplit=1, flags=re.MULTILINE | re.IGNORECASE,
    )[0]

    text = re.sub(r"```[\s\S]*?```", "", text)

    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s|:\-]+$", "", text, flags=re.MULTILINE)

    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)

    text = re.sub(r"^#{1,6}\s*\**(.+?)\**\s*$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    text = re.sub(r"^\s*[\*\-]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# SRT subtitle generation
# ---------------------------------------------------------------------------

def _fmt_srt_time(ticks: int) -> str:
    """Convert 100-nanosecond ticks to SRT time format HH:MM:SS,mmm."""
    ms = ticks // 10_000
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_eta(secs: float) -> str:
    s = int(secs)
    if s < 3600:
        return f"{s // 60}:{s % 60:02d}"
    h, rem = divmod(s, 3600)
    return f"{h}:{rem // 60:02d}:{rem % 60:02d}"


def _split_at_breaks(text: str) -> list[str]:
    """Split long text at clause-break punctuation, targeting _SRT_SPLIT_TARGET chars per chunk."""
    breaks = frozenset("，、；,;:：—")
    chunks: list[str] = []
    buf = ""

    for char in text:
        buf += char
        if len(buf) >= _SRT_SPLIT_TARGET and char in breaks:
            chunks.append(buf)
            buf = ""

    if buf:
        if chunks and len(buf) < _SRT_MIN_CHARS:
            chunks[-1] += buf
        else:
            chunks.append(buf)

    result: list[str] = []
    for chunk in (chunks or [text]):
        while len(chunk) > _SRT_MAX_CHARS * 2:
            result.append(chunk[:_SRT_MAX_CHARS])
            chunk = chunk[_SRT_MAX_CHARS:]
        result.append(chunk)
    return result


def build_quality_srt(boundaries: list[dict]) -> str:
    """Build well-segmented SRT from TTS boundary events.

    Each boundary (sentence or word) becomes one or more subtitle blocks.
    Long blocks are split at clause-break punctuation; short ones are
    merged with the next block to keep subtitle length moderate.
    """
    if not boundaries:
        return ""

    raw: list[tuple[int, int, str]] = []
    for b in boundaries:
        text = b["text"].strip()
        if not text:
            continue
        start, dur = b["offset"], b["duration"]
        if len(text) <= _SRT_MAX_CHARS:
            raw.append((start, start + dur, text))
        else:
            chunks = _split_at_breaks(text)
            total = len(text)
            pos = 0
            for chunk in chunks:
                t0 = start + int(dur * pos / total)
                pos += len(chunk)
                t1 = start + int(dur * pos / total)
                if chunk.strip():
                    raw.append((t0, t1, chunk.strip()))

    merged: list[tuple[int, int, str]] = []
    for s, e, text in raw:
        if merged and len(merged[-1][2]) < _SRT_MIN_CHARS:
            ps, _, pt = merged[-1]
            merged[-1] = (ps, e, pt + text)
        else:
            merged.append((s, e, text))

    parts: list[str] = []
    for idx, (s, e, text) in enumerate(merged, 1):
        parts.append(f"{idx}\n{_fmt_srt_time(s)} --> {_fmt_srt_time(e)}\n{text}\n")
    return "\n".join(parts)


def _write_srt(srt_path: Path, content: str) -> Path | None:
    """Atomically replace *srt_path* with this run's subtitles, if any."""
    if not content:
        srt_path.unlink(missing_ok=True)
        return None

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{srt_path.stem}.", suffix=f"{srt_path.suffix}.partial", dir=srt_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as srt_file:
            srt_file.write(content)
        temporary.replace(srt_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return srt_path


# ---------------------------------------------------------------------------
# TTS synthesis
# ---------------------------------------------------------------------------

def synthesize(text: str, voice: str, speed: float, output: Path) -> list[dict]:
    """Synthesize audio via edge-tts, returning word-boundary timing data."""
    rate = f"{int((speed - 1) * 100):+d}%"
    comm = edge_tts.Communicate(text, voice, rate=rate)

    total = len(text)
    chars = 0
    written = 0
    pct = -1
    bounds: list[dict] = []
    t0 = time.monotonic()

    with open(output, "wb") as f:
        for chunk in comm.stream_sync():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
                written += len(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                bounds.append({
                    "offset": chunk["offset"],
                    "duration": chunk["duration"],
                    "text": chunk.get("text", ""),
                })
                chars += len(chunk.get("text", ""))
                p = min(int(chars / total * 100), 99) if total else 0
                if p > pct:
                    pct = p
                    mb = written / (1024 * 1024)
                    elapsed = time.monotonic() - t0
                    eta = elapsed / pct * (100 - pct) if pct > 0 else 0
                    eta_s = f"~{_fmt_eta(eta)}" if pct > 0 else "?"
                    line = f"  [{pct:3d}%] {mb:.1f} MB  {_fmt_eta(elapsed)} / {eta_s}"
                    print(f"\r{line:<50}", end="", flush=True)

    elapsed = time.monotonic() - t0
    line = f"  [100%] {written / (1024 * 1024):.1f} MB  total {_fmt_eta(elapsed)}"
    print(f"\r{line:<50}")
    return bounds


async def _collect_dialogue_stream(communication: edge_tts.Communicate) -> tuple[bytes, list[dict]]:
    """Collect one Edge stream while allowing its async exceptions to propagate."""
    audio_parts: list[bytes] = []
    boundaries: list[dict] = []
    async for chunk in communication.stream():
        if chunk["type"] == "audio":
            audio_parts.append(chunk["data"])
        elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
            boundaries.append({
                "offset": chunk["offset"],
                "duration": chunk["duration"],
                "text": chunk.get("text", ""),
            })
    return b"".join(audio_parts), boundaries


def synthesize_dialogue_segment(text: str, voice: str, speed: float) -> tuple[bytes, list[dict]]:
    """Synthesize one dialogue turn with exception propagation and a hard timeout."""
    rate = f"{int((speed - 1) * 100):+d}%"
    communication = edge_tts.Communicate(text, voice, rate=rate)
    return asyncio.run(asyncio.wait_for(
        _collect_dialogue_stream(communication), timeout=_DIALOGUE_TURN_TIMEOUT_SECONDS,
    ))


def _mp3_duration_ticks(audio: bytes) -> int:
    """Estimate MP3 duration from frame headers without requiring ffmpeg."""
    if audio.startswith(b"ID3") and len(audio) >= 10:
        tag_size = sum((byte & 0x7F) << (7 * (3 - index)) for index, byte in enumerate(audio[6:10]))
        position = min(10 + tag_size, len(audio))
    else:
        position = 0

    mpeg1_rates = (44_100, 48_000, 32_000)
    mpeg2_rates = (22_050, 24_000, 16_000)
    mpeg25_rates = (11_025, 12_000, 8_000)
    mpeg1_layer3_bitrates = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
    mpeg2_layer3_bitrates = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
    seconds = 0.0
    while position + 4 <= len(audio):
        header = int.from_bytes(audio[position:position + 4], "big")
        if header >> 21 != 0x7FF:
            position += 1
            continue
        version = (header >> 19) & 0b11
        layer = (header >> 17) & 0b11
        bitrate_index = (header >> 12) & 0b1111
        sample_rate_index = (header >> 10) & 0b11
        padding = (header >> 9) & 0b1
        if version == 1 or layer != 1 or bitrate_index in (0, 15) or sample_rate_index == 3:
            position += 1
            continue
        if version == 3:
            sample_rate = mpeg1_rates[sample_rate_index]
            bitrate = mpeg1_layer3_bitrates[bitrate_index]
            samples_per_frame = 1_152
            frame_size = 144 * bitrate * 1_000 // sample_rate + padding
        else:
            sample_rate = (mpeg2_rates if version == 2 else mpeg25_rates)[sample_rate_index]
            bitrate = mpeg2_layer3_bitrates[bitrate_index]
            samples_per_frame = 576
            frame_size = 72 * bitrate * 1_000 // sample_rate + padding
        if frame_size < 4 or position + frame_size > len(audio):
            position += 1
            continue
        seconds += samples_per_frame / sample_rate
        position += frame_size
    return round(seconds * _TICKS_PER_SECOND)


def _dialogue_segment_duration(boundaries: list[dict], audio: bytes) -> int:
    """Use the MP3 duration, falling back to the final boundary when unavailable."""
    audio_duration = _mp3_duration_ticks(audio)
    if audio_duration:
        return audio_duration

    boundary_duration = max(
        (int(boundary["offset"]) + int(boundary["duration"]) for boundary in boundaries),
        default=0,
    )
    if boundary_duration:
        return boundary_duration
    warnings.warn(
        "Dialogue turn had no usable timing data; later subtitle offsets may be inaccurate.",
        UserWarning,
        stacklevel=2,
    )
    return 0


def _dialogue_cache_dir() -> Path:
    """Return the XDG cache directory shared by resumable dialogue turns."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    cache_home = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"
    return cache_home / "md2audio" / "dialogue" / "turns"


def _dialogue_turn_identity(turn: DialogueTurn, voice: str, speed: float) -> dict[str, object]:
    """Build the stable metadata used to validate one cached dialogue turn."""
    return {
        "speaker": turn.speaker,
        "voice": voice,
        "speed": float(speed),
        "text_sha256": hashlib.sha256(turn.text.encode("utf-8")).hexdigest(),
    }


def _dialogue_turn_manifest_path(cache_dir: Path, identity: dict[str, object]) -> Path:
    """Return the atomically published cache manifest path for one turn identity."""
    canonical_identity = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()
    return cache_dir / f"{cache_key}.json"


def _dialogue_audio_cache_path(cache_dir: Path, audio_sha256: str) -> Path:
    """Return the immutable content-addressed audio path named by its checksum."""
    return cache_dir / f"{audio_sha256}.mp3"


def _load_cached_dialogue_turn(
    cache_dir: Path, identity: dict[str, object],
) -> tuple[bytes, list[dict]] | None:
    """Load one valid cached turn, ignoring incomplete or corrupt cache entries."""
    try:
        metadata = json.loads(_dialogue_turn_manifest_path(cache_dir, identity).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    boundaries = metadata.get("boundaries")
    audio_sha256 = metadata.get("audio_sha256")
    if (
        metadata.get("schema_version") != _DIALOGUE_CACHE_SCHEMA_VERSION
        or metadata.get("identity") != identity
        or not isinstance(audio_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", audio_sha256) is None
        or not isinstance(boundaries, list)
        or not all(isinstance(boundary, dict) for boundary in boundaries)
    ):
        return None
    try:
        audio = _dialogue_audio_cache_path(cache_dir, audio_sha256).read_bytes()
    except OSError:
        return None
    if hashlib.sha256(audio).hexdigest() != audio_sha256:
        return None
    return audio, boundaries


def _write_cached_dialogue_turn(
    cache_dir: Path, identity: dict[str, object], audio: bytes, boundaries: list[dict],
) -> None:
    """Store immutable audio, then atomically publish the matching turn manifest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_sha256 = hashlib.sha256(audio).hexdigest()
    audio_path = _dialogue_audio_cache_path(cache_dir, audio_sha256)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{audio_path.stem}.", suffix=".mp3.partial", dir=cache_dir,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as audio_file:
            audio_file.write(audio)
        temporary.replace(audio_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    metadata = {
        "schema_version": _DIALOGUE_CACHE_SCHEMA_VERSION,
        "identity": identity,
        "audio_sha256": audio_sha256,
        "boundaries": boundaries,
    }
    metadata_path = _dialogue_turn_manifest_path(cache_dir, identity)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{metadata_path.stem}.", suffix=".json.partial", dir=cache_dir,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, sort_keys=True)
        temporary.replace(metadata_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _is_retryable_dialogue_error(error: Exception) -> bool:
    """Return whether an Edge/network failure is safe to retry on this turn."""
    if isinstance(error, PermissionError):
        return False
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, getattr(asyncio, "TimeoutError", ())):
        return True
    if isinstance(error, (ConnectionError, OSError)):
        return True
    error_type = type(error)
    if (
        error_type.__module__.startswith("aiohttp.")
        and error_type.__name__ in _AIOHTTP_RETRYABLE_ERROR_NAMES
    ):
        return True
    if (
        error_type.__module__ == "edge_tts.exceptions"
        and error_type.__name__ == "WebSocketError"
    ):
        return True
    status = getattr(error, "status", None)
    if status is None:
        status = getattr(error, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    return bool(re.search(r"\b(?:429|5\d\d)\b", str(error)))


def _synthesize_dialogue_turn_with_retries(
    text: str, voice: str, speed: float,
) -> tuple[bytes, list[dict]]:
    """Synthesize one turn with bounded exponential backoff for transient errors."""
    for attempt in range(1, _DIALOGUE_RETRY_ATTEMPTS + 1):
        try:
            return synthesize_dialogue_segment(text, voice, speed)
        except Exception as exc:
            if not _is_retryable_dialogue_error(exc) or attempt == _DIALOGUE_RETRY_ATTEMPTS:
                raise
            delay = min(2 ** (attempt - 1), _DIALOGUE_MAX_RETRY_DELAY_SECONDS)
            print(
                f"    transient failure ({exc}); retrying in {delay}s "
                f"({attempt}/{_DIALOGUE_RETRY_ATTEMPTS})"
            )
            time.sleep(delay)
    raise AssertionError("dialogue retry loop exhausted without returning or raising")


def _prepare_dialogue_cache_dir(cache_dir: Path) -> Path | None:
    """Return a writable cache directory, or disable caching before any network work."""
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".md2audio-write-check.", dir=cache_dir)
        temporary = Path(temporary_name)
        os.close(descriptor)
        descriptor = None
        temporary.unlink()
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        warnings.warn(
            f"Dialogue cache is unavailable at {cache_dir}: {exc}; continuing without cache. "
            "Use --no-cache to silence this warning.",
            UserWarning,
            stacklevel=2,
        )
        return None
    return cache_dir


def synthesize_dialogue(
    turns: list[DialogueTurn],
    voice_map: dict[str, str],
    speed: float,
    output: Path,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> list[dict]:
    """Atomically synthesize, concatenate, and offset a resumable dialogue.

    Edge TTS returns identically encoded MP3 frames for each request, so the
    completed output is a sequential binary concatenation. The original output
    is retained unless every turn succeeds. Successful turns are cached by
    their text, speaker, voice, and speed unless ``use_cache`` is false.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    turn_cache_dir = cache_dir or _dialogue_cache_dir() if use_cache else None
    if turn_cache_dir is not None:
        turn_cache_dir = _prepare_dialogue_cache_dir(turn_cache_dir)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=f"{output.suffix}.partial", dir=output.parent,
    )
    temporary = Path(temporary_name)
    cumulative_duration = 0
    all_boundaries: list[dict] = []
    try:
        with os.fdopen(descriptor, "wb") as combined_audio:
            for index, turn in enumerate(turns, 1):
                voice = voice_map[turn.speaker]
                identity = _dialogue_turn_identity(turn, voice, speed)
                cached_turn = (
                    _load_cached_dialogue_turn(turn_cache_dir, identity)
                    if turn_cache_dir is not None else None
                )
                if cached_turn is not None:
                    audio, boundaries = cached_turn
                    print(f"  Turn {index}/{len(turns)}: {turn.speaker} ({voice}) [cache hit]")
                else:
                    print(f"  Turn {index}/{len(turns)}: {turn.speaker} ({voice}) [synthesizing]")
                    try:
                        audio, boundaries = _synthesize_dialogue_turn_with_retries(
                            turn.text, voice, speed,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Synthesis failed at turn {index} for speaker {turn.speaker!r} "
                            f"with voice {voice!r}: {exc}"
                        ) from exc
                    if turn_cache_dir is not None:
                        try:
                            _write_cached_dialogue_turn(turn_cache_dir, identity, audio, boundaries)
                        except OSError as exc:
                            warnings.warn(
                                f"Could not write dialogue cache at {turn_cache_dir}: {exc}; "
                                "continuing without cache. Use --no-cache to silence this warning.",
                                UserWarning,
                                stacklevel=2,
                            )
                            turn_cache_dir = None
                combined_audio.write(audio)
                for boundary in boundaries:
                    shifted = dict(boundary)
                    shifted["offset"] = int(shifted["offset"]) + cumulative_duration
                    all_boundaries.append(shifted)
                cumulative_duration += _dialogue_segment_duration(boundaries, audio)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return all_boundaries


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

_CODEC_NVENC = ["-c:v", "hevc_nvenc", "-b:v", "0", "-cq", "30", "-preset", "p6"]
_CODEC_X264 = ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]


def _nvenc_available() -> bool:
    """Check if hevc_nvenc encoder is listed by ffmpeg."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        return "hevc_nvenc" in r.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def generate_video(audio: Path, video: Path, srt: Path | None = None) -> None:
    """Generate waveform video from audio, optionally burning in subtitles."""
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found", file=sys.stderr)
        sys.exit(1)

    filt = (
        "[0:a]showwaves=s=1280x720:mode=cline:rate=30"
        ":colors=0x4a90d9@0.8|0x7ec8e3@0.6:scale=sqrt[waves];"
        "color=c=0x1a1a2e:s=1280x720:r=30[bg];"
        "[bg][waves]overlay=format=auto,format=yuv420p[v]"
    )

    codecs_to_try = [_CODEC_NVENC, _CODEC_X264] if _nvenc_available() else [_CODEC_X264]

    for codec in codecs_to_try:
        cmd = [
            "ffmpeg", "-i", str(audio),
            "-filter_complex", filt,
            "-map", "[v]", "-map", "0:a",
            *codec,
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-y", str(video),
        ]
        print(f"  Encoding video ({codec[1]})...")
        t0_enc = time.monotonic()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  Encoded in {_fmt_eta(time.monotonic() - t0_enc)}")
            break
    else:
        print(f"ffmpeg error:\n{r.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)

    if srt and srt.exists():
        print("  Burning subtitles...")
        t0_sub = time.monotonic()
        tmp = video.with_suffix(".tmp.mp4")
        sub_cmd = [
            "ffmpeg", "-i", str(video),
            "-vf", f"subtitles={srt}:force_style='FontSize=24,PrimaryColour=&Hffffff&'",
            "-c:a", "copy", "-y", str(tmp),
        ]
        r = subprocess.run(sub_cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  Subtitles burned in {_fmt_eta(time.monotonic() - t0_sub)}")
            tmp.replace(video)
        else:
            print("  Warning: could not burn subtitles into video", file=sys.stderr)
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    """Register arguments shared by synth and bundle subcommands."""
    p.add_argument("input", type=Path, help="Input file (markdown or plain text)")
    p.add_argument("-o", "--output", type=Path,
                   help="Output audio file (default: ~/w/<stem>.mp3)")
    g = p.add_argument_group("TTS")
    g.add_argument("--voice",
                   help=f"Voice (default: {DEFAULT_VOICE})")
    g.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                   help=f"Speed multiplier (default: {DEFAULT_SPEED})")
    g = p.add_argument_group("Processing")
    g.add_argument("--no-strip", action="store_true",
                   help="Don't strip markdown formatting")
    g = p.add_argument_group("Multi-speaker dialogue")
    g.add_argument("--multi-speaker", action="store_true",
                   help="Parse line-start bold speaker labels and synthesize each turn")
    g.add_argument("--speaker-gender", choices=("male", "female"),
                   help="Automatic mode: required Edge voice gender")
    selectors = g.add_mutually_exclusive_group()
    selectors.add_argument("--speaker-locale",
                           help="Automatic mode: exact Edge voice locale, for example zh-CN")
    selectors.add_argument("--speaker-language",
                           help="Automatic mode: voice language, for example zh")
    g.add_argument("--seed", type=int,
                   help="Automatic mode: deterministic speaker-to-voice assignment seed")
    g.add_argument("--speaker-config", type=Path,
                   help="TOML file mapping each speaker label to an Edge voice")
    g.add_argument("--no-cache", action="store_true",
                   help="Multi-speaker mode: synthesize every turn without reading or writing cache")


def _resolve(args) -> tuple[Path, str]:
    """Validate input, resolve output path, read and process text."""
    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)
    out = args.output or Path.home() / "w" / f"{args.input.stem}.mp3"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = args.input.read_text(encoding="utf-8")
    text = raw if args.no_strip else strip_markdown(raw)
    print(f"Input:  {args.input} ({len(raw)} chars → {len(text)} after processing)")
    return out, text


def _resolve_dialogue(args) -> tuple[Path, list[DialogueTurn]]:
    """Read, parse, and optionally strip a multi-speaker input document."""
    if not args.input.exists():
        raise ValueError(f"Input file not found: {args.input}")
    out = args.output or Path.home() / "w" / f"{args.input.stem}.mp3"
    raw = args.input.read_text(encoding="utf-8")
    parsed_turns = parse_dialogue(raw)
    turns: list[DialogueTurn] = []
    for turn in parsed_turns:
        text = turn.text if args.no_strip else strip_markdown(turn.text)
        if not text:
            warnings.warn(
                f"Ignoring empty dialogue turn for speaker {turn.speaker!r} after Markdown processing.",
                UserWarning,
                stacklevel=2,
            )
            continue
        turns.append(DialogueTurn(turn.speaker, text))
    if not turns:
        raise ValueError("Multi-speaker input contains no speakable dialogue turns.")
    print(
        f"Input:  {args.input} ({len(raw)} chars → "
        f"{sum(len(turn.text) for turn in turns)} across {len(turns)} turns)"
    )
    return out, turns


def _resolve_dialogue_voices(args, speakers: list[str]) -> dict[str, str]:
    """Choose dialogue voices from either strict TOML config or Edge metadata."""
    if args.speaker_config:
        config = load_speaker_config(args.speaker_config)
        try:
            voices = fetch_voices()
        except Exception as exc:
            warnings.warn(
                "Could not fetch the Edge voice list; proceeding with speaker config without "
                f"availability validation: {exc}",
                UserWarning,
                stacklevel=2,
            )
            voices = None
        return resolve_config_voice_map(speakers, config, voices)
    try:
        voices = fetch_voices()
    except Exception as exc:
        raise ValueError(
            f"Could not fetch the Edge voice list required for automatic speaker assignment: {exc}"
        ) from exc
    return resolve_auto_voice_map(
        speakers,
        voices,
        args.speaker_gender,
        locale=args.speaker_locale,
        language=args.speaker_language,
        seed=args.seed,
    )


def _prepare_multi_speaker(args) -> tuple[Path, list[DialogueTurn], dict[str, str]]:
    """Resolve all offline input and one-time voice selection before synthesis."""
    out, turns = _resolve_dialogue(args)
    voice_map = _resolve_dialogue_voices(
        args, list(dict.fromkeys(turn.speaker for turn in turns))
    )
    print("Speaker voices:")
    for speaker, voice in voice_map.items():
        print(f"  {speaker}: {voice}")
    return out, turns, voice_map


def _validate_multi_speaker_args(args, parser: argparse.ArgumentParser) -> None:
    """Reject dialogue selectors that are incomplete or would be ignored."""
    has_dialogue_selector = any((
        args.speaker_gender,
        args.speaker_locale,
        args.speaker_language,
        args.speaker_config,
        args.seed is not None,
        args.no_cache,
    ))
    if not args.multi_speaker:
        if has_dialogue_selector:
            parser.error("--speaker-* options, --seed, and --no-cache require --multi-speaker")
        return
    if args.voice is not None:
        parser.error("--voice and --multi-speaker are mutually exclusive")
    if args.speaker_config:
        if any((args.speaker_gender, args.speaker_locale, args.speaker_language, args.seed is not None)):
            parser.error("--speaker-config cannot be combined with automatic speaker selectors")
        return
    if not args.speaker_gender:
        parser.error("automatic multi-speaker mode requires --speaker-gender")
    if bool(args.speaker_locale) == bool(args.speaker_language):
        parser.error("automatic multi-speaker mode needs exactly one of --speaker-locale or --speaker-language")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _cmd_synth(args) -> None:
    """Synthesize audio, with optional SRT and video."""
    if args.multi_speaker:
        try:
            out, turns, voice_map = _prepare_multi_speaker(args)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        srt_out = out.with_suffix(".srt") if args.srt else None

        print(f"Output: {out}")
        print("Synthesizing dialogue..." + (" (with subtitles)" if srt_out else ""))
        try:
            bounds = synthesize_dialogue(
                turns, voice_map, args.speed, out, use_cache=not args.no_cache,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Audio:  {out.stat().st_size / (1024 * 1024):.1f} MB")

        if srt_out:
            srt_content = build_quality_srt(bounds)
            srt_out = _write_srt(srt_out, srt_content)
            if srt_out:
                print(f"SRT:    {srt_content.count(' --> ')} blocks → {srt_out}")

        if args.video:
            vp = out.with_suffix(".mp4")
            generate_video(out, vp, srt_out if srt_out and srt_out.exists() else None)
            if vp.exists():
                print(f"Video:  {vp.stat().st_size / (1024 * 1024):.1f} MB → {vp}")

        print("Done!")
        return

    out, text = _resolve(args)
    srt_out = out.with_suffix(".srt") if args.srt else None

    voice = args.voice or DEFAULT_VOICE
    print(f"Voice:  {voice} @ {args.speed}x")
    print(f"Output: {out}")
    print("Synthesizing..." + (" (with subtitles)" if srt_out else ""))

    bounds = synthesize(text, voice, args.speed, out)
    print(f"Audio:  {out.stat().st_size / (1024 * 1024):.1f} MB")

    if srt_out:
        srt_content = build_quality_srt(bounds)
        srt_out = _write_srt(srt_out, srt_content)
        if srt_out:
            print(f"SRT:    {srt_content.count(' --> ')} blocks → {srt_out}")

    if args.video:
        vp = out.with_suffix(".mp4")
        generate_video(out, vp, srt_out if srt_out and srt_out.exists() else None)
        if vp.exists():
            print(f"Video:  {vp.stat().st_size / (1024 * 1024):.1f} MB → {vp}")

    print("Done!")


def _cmd_bundle(args) -> None:
    """Generate audio + SRT subtitles + waveform video in one step."""
    if args.multi_speaker:
        try:
            out, turns, voice_map = _prepare_multi_speaker(args)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        srt_path = out.with_suffix(".srt")
        vid_path = out.with_suffix(".mp4")

        print(f"Audio:  {out}")
        print(f"SRT:    {srt_path}")
        print(f"Video:  {vid_path}")

        print("\n[1/3] Synthesizing dialogue...")
        try:
            bounds = synthesize_dialogue(
                turns, voice_map, args.speed, out, use_cache=not args.no_cache,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"  {out.stat().st_size / (1024 * 1024):.1f} MB")

        print("\n[2/3] Generating subtitles...")
        srt_content = build_quality_srt(bounds)
        srt_path = _write_srt(srt_path, srt_content)
        if srt_path:
            print(f"  {srt_content.count(' --> ')} subtitle blocks")
        else:
            print("  Warning: no boundary data from TTS", file=sys.stderr)

        print("\n[3/3] Generating video...")
        generate_video(out, vid_path, srt_path)
        if vid_path.exists():
            print(f"  {vid_path.stat().st_size / (1024 * 1024):.1f} MB")

        print("\nDone!")
        return

    out, text = _resolve(args)
    srt_path = out.with_suffix(".srt")
    vid_path = out.with_suffix(".mp4")

    voice = args.voice or DEFAULT_VOICE
    print(f"Voice:  {voice} @ {args.speed}x")
    print(f"Audio:  {out}")
    print(f"SRT:    {srt_path}")
    print(f"Video:  {vid_path}")

    print("\n[1/3] Synthesizing audio...")
    bounds = synthesize(text, voice, args.speed, out)
    print(f"  {out.stat().st_size / (1024 * 1024):.1f} MB")

    print("\n[2/3] Generating subtitles...")
    srt_content = build_quality_srt(bounds)
    srt_path = _write_srt(srt_path, srt_content)
    if srt_path:
        print(f"  {srt_content.count(' --> ')} subtitle blocks")
    else:
        print("  Warning: no boundary data from TTS", file=sys.stderr)

    print("\n[3/3] Generating video...")
    generate_video(out, vid_path, srt_path)
    if vid_path.exists():
        print(f"  {vid_path.stat().st_size / (1024 * 1024):.1f} MB")

    print("\nDone!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _commands = {"synth", "bundle"}
    if len(sys.argv) > 1 and sys.argv[1] not in _commands | {"-h", "--help"}:
        sys.argv.insert(1, "synth")

    top = argparse.ArgumentParser(
        prog="md2audio",
        description="Convert markdown/text to audio, subtitles, and video.",
    )
    sub = top.add_subparsers(dest="command")

    sp = sub.add_parser("synth", help="Synthesize audio (default when omitted)")
    _add_common(sp)
    g = sp.add_argument_group("Extras")
    g.add_argument("--srt", action="store_true", help="Also generate SRT subtitles")
    g.add_argument("--video", action="store_true", help="Also generate MP4 video")

    bp = sub.add_parser("bundle", help="Generate audio + SRT + video in one step")
    _add_common(bp)

    args = top.parse_args()
    if args.command is not None:
        _validate_multi_speaker_args(args, top)
    if args.command == "synth":
        _cmd_synth(args)
    elif args.command == "bundle":
        _cmd_bundle(args)
    else:
        top.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
