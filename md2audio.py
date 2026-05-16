"""Convert markdown files to audio (and optionally subtitles/video) via edge-tts."""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import edge_tts

DEFAULT_VOICE = "zh-CN-YunyangNeural"
DEFAULT_SPEED = 1.0
_SRT_MAX_CHARS = 35
_SRT_MIN_CHARS = 10
_SRT_SPLIT_TARGET = 20


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

    text = re.sub(r"\s+\d+(?:(?:,\s*|\s*)\d+)*(?=[\s。，,.;；])", "", text)

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
                    print(f"\r  [{pct:3d}%] {mb:.1f} MB", end="", flush=True)

    print(f"\r  [100%] {written / (1024 * 1024):.1f} MB")
    return bounds


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def _pick_codec() -> list[str]:
    """Prefer NVENC hardware encoding; fall back to libx264."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        if "hevc_nvenc" in r.stdout:
            return ["-c:v", "hevc_nvenc", "-b:v", "0", "-cq", "30", "-preset", "p6"]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]


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

    cmd = [
        "ffmpeg", "-i", str(audio),
        "-filter_complex", filt,
        "-map", "[v]", "-map", "0:a",
        *_pick_codec(),
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-y", str(video),
    ]

    print("  Encoding video...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ffmpeg error:\n{r.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)

    if srt and srt.exists():
        tmp = video.with_suffix(".tmp.mp4")
        sub_cmd = [
            "ffmpeg", "-i", str(video),
            "-vf", f"subtitles={srt}:force_style='FontSize=24,PrimaryColour=&Hffffff&'",
            "-c:a", "copy", "-y", str(tmp),
        ]
        r = subprocess.run(sub_cmd, capture_output=True, text=True)
        if r.returncode == 0:
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


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _cmd_synth(args) -> None:
    """Synthesize audio, with optional SRT and video."""
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
    if len(sys.argv) > 1 and sys.argv[1] not in _commands and not sys.argv[1].startswith("-"):
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
    if args.command == "synth":
        _cmd_synth(args)
    elif args.command == "bundle":
        _cmd_bundle(args)
    else:
        top.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
