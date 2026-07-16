"""Convert markdown files to audio (and optionally subtitles/video) via edge-tts."""

import argparse
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


def synthesize_dialogue_segment(text: str, voice: str, speed: float) -> tuple[bytes, list[dict]]:
    """Synthesize one dialogue turn into MP3 bytes and its local boundaries."""
    rate = f"{int((speed - 1) * 100):+d}%"
    communication = edge_tts.Communicate(text, voice, rate=rate)
    audio_parts: list[bytes] = []
    boundaries: list[dict] = []
    for chunk in communication.stream_sync():
        if chunk["type"] == "audio":
            audio_parts.append(chunk["data"])
        elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
            boundaries.append({
                "offset": chunk["offset"],
                "duration": chunk["duration"],
                "text": chunk.get("text", ""),
            })
    return b"".join(audio_parts), boundaries


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
    """Prefer boundary timing, falling back to MP3 frame duration when needed."""
    boundary_duration = max(
        (int(boundary["offset"]) + int(boundary["duration"]) for boundary in boundaries),
        default=0,
    )
    if boundary_duration:
        return boundary_duration
    audio_duration = _mp3_duration_ticks(audio)
    if audio_duration:
        warnings.warn(
            "Dialogue turn had no boundary events; using MP3 frame duration for subtitle offsets.",
            UserWarning,
            stacklevel=2,
        )
    else:
        warnings.warn(
            "Dialogue turn had no usable timing data; later subtitle offsets may be inaccurate.",
            UserWarning,
            stacklevel=2,
        )
    return audio_duration


def synthesize_dialogue(
    turns: list[DialogueTurn], voice_map: dict[str, str], speed: float, output: Path,
) -> list[dict]:
    """Atomically synthesize, concatenate, and offset a multi-speaker dialogue.

    Edge TTS returns identically encoded MP3 frames for each request, so the
    completed output is a sequential binary concatenation. The original output
    is retained unless every turn succeeds.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
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
                print(f"  Turn {index}/{len(turns)}: {turn.speaker} ({voice})")
                try:
                    audio, boundaries = synthesize_dialogue_segment(turn.text, voice, speed)
                except Exception as exc:
                    raise RuntimeError(
                        f"Synthesis failed at turn {index} for speaker {turn.speaker!r} "
                        f"with voice {voice!r}: {exc}"
                    ) from exc
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
    g.add_argument("--voice", default=DEFAULT_VOICE,
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
    voices = fetch_voices()
    if args.speaker_config:
        return resolve_config_voice_map(speakers, load_speaker_config(args.speaker_config), voices)
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
    ))
    if not args.multi_speaker:
        if has_dialogue_selector:
            parser.error("--speaker-* options and --seed require --multi-speaker")
        return
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
            bounds = synthesize_dialogue(turns, voice_map, args.speed, out)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Audio:  {out.stat().st_size / (1024 * 1024):.1f} MB")

        if srt_out:
            srt_content = build_quality_srt(bounds)
            if srt_content:
                srt_out.write_text(srt_content, encoding="utf-8")
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

    print(f"Voice:  {args.voice} @ {args.speed}x")
    print(f"Output: {out}")
    print("Synthesizing..." + (" (with subtitles)" if srt_out else ""))

    bounds = synthesize(text, args.voice, args.speed, out)
    print(f"Audio:  {out.stat().st_size / (1024 * 1024):.1f} MB")

    if srt_out:
        srt_content = build_quality_srt(bounds)
        if srt_content:
            srt_out.write_text(srt_content, encoding="utf-8")
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
            bounds = synthesize_dialogue(turns, voice_map, args.speed, out)
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"  {out.stat().st_size / (1024 * 1024):.1f} MB")

        print("\n[2/3] Generating subtitles...")
        srt_content = build_quality_srt(bounds)
        if srt_content:
            srt_path.write_text(srt_content, encoding="utf-8")
            print(f"  {srt_content.count(' --> ')} subtitle blocks")
        else:
            print("  Warning: no boundary data from TTS", file=sys.stderr)
            srt_path = None

        print("\n[3/3] Generating video...")
        generate_video(out, vid_path, srt_path)
        if vid_path.exists():
            print(f"  {vid_path.stat().st_size / (1024 * 1024):.1f} MB")

        print("\nDone!")
        return

    out, text = _resolve(args)
    srt_path = out.with_suffix(".srt")
    vid_path = out.with_suffix(".mp4")

    print(f"Voice:  {args.voice} @ {args.speed}x")
    print(f"Audio:  {out}")
    print(f"SRT:    {srt_path}")
    print(f"Video:  {vid_path}")

    print("\n[1/3] Synthesizing audio...")
    bounds = synthesize(text, args.voice, args.speed, out)
    print(f"  {out.stat().st_size / (1024 * 1024):.1f} MB")

    print("\n[2/3] Generating subtitles...")
    srt_content = build_quality_srt(bounds)
    if srt_content:
        srt_path.write_text(srt_content, encoding="utf-8")
        print(f"  {srt_content.count(' --> ')} subtitle blocks")
    else:
        print("  Warning: no boundary data from TTS", file=sys.stderr)
        srt_path = None

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
