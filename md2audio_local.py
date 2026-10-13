"""Offline VITS synthesis through the native sherpa-onnx CLI and FFmpeg."""

import math
import os
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
import wave
from contextlib import ExitStack
from pathlib import Path

__all__ = ["LocalTTSError", "synthesize_local"]

_PROCESS_TIMEOUT_SECONDS = 300
# shortcut: the CLI reloads the model per chunk; use a persistent runtime if long-document latency becomes limiting.
_MAX_CHUNK_CHARS = 400
_TICKS_PER_SECOND = 10_000_000


class LocalTTSError(RuntimeError):
    """A local dependency, model, or synthesis process could not complete."""


def _run_process(command: list[str]) -> None:
    """Drain bounded diagnostic output and kill/reap the process group on failure."""
    tail = b""
    try:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   start_new_session=True)
    except OSError as exc:
        raise LocalTTSError(f"Cannot execute {command[0]}: {exc}") from exc
    try:
        with selectors.DefaultSelector() as selector:
            if process.stderr is None:
                raise LocalTTSError("Native process did not provide a diagnostic pipe")
            selector.register(process.stderr, selectors.EVENT_READ)
            os.set_blocking(process.stderr.fileno(), False)
            deadline = time.monotonic() + _PROCESS_TIMEOUT_SECONDS
            while selector.get_map() or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LocalTTSError(f"{Path(command[0]).name} timed out after {_PROCESS_TIMEOUT_SECONDS}s")
                for key, _ in selector.select(min(remaining, 0.1)):
                    data = os.read(key.fd, 8192)
                    if data:
                        tail = (tail + data)[-2048:]
                    else:
                        selector.unregister(key.fileobj)
            returncode = process.wait()
        if returncode:
            raise LocalTTSError(f"{Path(command[0]).name} exited {returncode}: "
                                f"{tail.decode('utf-8', errors='replace').strip()}")
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    finally:
        if process.stderr is not None:
            process.stderr.close()


def _text_chunks(text: str) -> list[str]:
    """Bound argv size and inference memory, preferring natural sentence breaks."""
    chunks: list[str] = []
    pending = ""
    for sentence in re.findall(r"[^。！？.!?\n]+[。！？.!?\n]*|[。！？.!?\n]+", text):
        while len(sentence) > _MAX_CHUNK_CHARS:
            if pending:
                chunks.append(pending)
                pending = ""
            chunks.append(sentence[:_MAX_CHUNK_CHARS])
            sentence = sentence[_MAX_CHUNK_CHARS:]
        if len(pending) + len(sentence) > _MAX_CHUNK_CHARS:
            chunks.append(pending)
            pending = ""
        pending += sentence
    if pending:
        chunks.append(pending)
    return [chunk for chunk in chunks if chunk.strip()]


def synthesize_local(
    text: str, model_dir: Path, speaker: str, speed: float, output: Path, volume: str = "2.00",
) -> list[dict]:
    """Atomically produce offline MP3 with one gain and measured chunk subtitle timing.

    ``model_dir`` contains model.onnx, lexicon.txt, and tokens.txt. A speakers.txt
    sidecar lists speaker names in zero-based ID order; without it only ID 0 is
    accepted. sherpa-onnx-offline-tts and ffmpeg must be installed on PATH.
    Each bounded text chunk has a 300-second process deadline. Subtitle timing
    is exact at chunk boundaries; intra-chunk timing is estimated by the caller.
    The existing output is retained on any failure, with no online fallback.
    """
    if re.fullmatch(r"[0-9]+", speaker) is None:
        raise LocalTTSError("Local speaker ID must be a nonnegative integer")
    if not math.isfinite(speed) or speed <= 0:
        raise LocalTTSError("Local speed must be finite and greater than zero")
    if re.fullmatch(r"[0-9]+(?:\.[0-9]{1,2})?", volume) is None:
        raise LocalTTSError("Local volume must be nonnegative with at most two decimal places")
    chunks = _text_chunks(text)
    if not chunks:
        raise LocalTTSError("Input contains no speakable text")
    model_dir = model_dir.expanduser()
    for filename in ("model.onnx", "lexicon.txt", "tokens.txt"):
        if not (model_dir / filename).is_file():
            raise LocalTTSError(f"Missing VITS model file: {model_dir / filename}")
    speakers_file = model_dir / "speakers.txt"
    try:
        speakers = speakers_file.read_text(encoding="utf-8").splitlines() if speakers_file.exists() else ["0"]
    except (OSError, UnicodeError) as exc:
        raise LocalTTSError(f"Cannot read speaker list {speakers_file}: {exc}") from exc
    if not speakers or any(not name.strip() for name in speakers):
        raise LocalTTSError(f"Invalid speaker list: {speakers_file}")
    try:
        sid = int(speaker)
    except ValueError as exc:
        raise LocalTTSError("Local speaker ID is too large to parse") from exc
    if sid >= len(speakers):
        raise LocalTTSError(f"Speaker ID {sid} is unavailable; this model supports 0–{len(speakers) - 1}")
    executables: list[str] = []
    for name in ("sherpa-onnx-offline-tts", "ffmpeg"):
        executable = shutil.which(name)
        if executable is None:
            raise LocalTTSError(f"{name} not found on PATH; install the native CLI before using --local-tts")
        executables.append(executable)
    output.parent.mkdir(parents=True, exist_ok=True)
    boundaries: list[dict] = []
    total_frames = 0
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output.stem}.local-", dir=output.parent) as directory:
            work = Path(directory)
            combined = work / "combined.wav"
            with ExitStack() as files:
                merged = None
                audio_format: tuple[int, int, int] | None = None
                for index, chunk in enumerate(chunks, 1):
                    print(f"  Local chunk {index}/{len(chunks)} (speaker {sid})", flush=True)
                    wav_path = work / "chunk.wav"
                    command = [executables[0], "--num-threads=2", f"--sid={sid}", f"--speed={speed}",
                               f"--vits-model={model_dir / 'model.onnx'}",
                               f"--vits-lexicon={model_dir / 'lexicon.txt'}",
                               f"--vits-tokens={model_dir / 'tokens.txt'}",
                               f"--output-filename={wav_path}"]
                    fsts = [str(model_dir / name) for name in ("phone.fst", "date.fst", "number.fst")
                            if (model_dir / name).is_file()]
                    if fsts:
                        command.append("--tts-rule-fsts=" + ",".join(fsts))
                    command.extend(["--", chunk])
                    _run_process(command)
                    with wave.open(str(wav_path), "rb") as source:
                        current_format = (source.getnchannels(), source.getsampwidth(), source.getframerate())
                        if source.getcomptype() != "NONE" or source.getnframes() == 0:
                            raise LocalTTSError("Native TTS produced empty or unsupported WAV audio")
                        if audio_format is None:
                            audio_format = current_format
                            merged = files.enter_context(wave.open(str(combined), "wb"))
                            merged.setparams((*current_format, 0, "NONE", "not compressed"))
                        elif current_format != audio_format:
                            raise LocalTTSError("Native TTS changed WAV format between chunks")
                        start = total_frames * _TICKS_PER_SECOND // source.getframerate()
                        total_frames += source.getnframes()
                        end = total_frames * _TICKS_PER_SECOND // source.getframerate()
                        boundaries.append({"text": chunk, "offset": start, "duration": end - start})
                        read_bytes = 0
                        while data := source.readframes(65536):
                            if merged is None:
                                raise LocalTTSError("Could not initialize combined WAV output")
                            read_bytes += len(data)
                            merged.writeframesraw(data)
                        if read_bytes != source.getnframes() * source.getnchannels() * source.getsampwidth():
                            raise LocalTTSError("Native TTS produced truncated WAV audio")
                    wav_path.unlink()
            mp3 = work / "audio.mp3"
            _run_process([executables[1], "-hide_banner", "-loglevel", "error", "-y", "-i", str(combined),
                          "-af", f"volume={volume}", "-c:a", "libmp3lame", "-q:a", "4", str(mp3)])
            if mp3.stat().st_size == 0:
                raise LocalTTSError("FFmpeg produced empty MP3 audio")
            mp3.replace(output)
    except (OSError, EOFError, wave.Error) as exc:
        raise LocalTTSError(f"Could not complete local audio output: {exc}") from exc
    return boundaries
