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


# ---------------------------------------------------------------------------
# Markdown stripping
# ---------------------------------------------------------------------------

def strip_markdown(text: str) -> str:
    """Remove markdown syntax, keeping readable plain text."""
    # Remove reference/citation sections and everything after
    text = re.split(
        r"^#{1,6}\s*\**(?:引用的著作|参考文献|References?|Bibliography)\**\s*$",
        text, maxsplit=1, flags=re.MULTILINE | re.IGNORECASE,
    )[0]

    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Remove markdown tables
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s|:\-]+$", "", text, flags=re.MULTILINE)

    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # Remove links, keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove bare URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove headings markers but keep text
    text = re.sub(r"^#{1,6}\s*\**(.+?)\**\s*$", r"\1", text, flags=re.MULTILINE)

    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)

    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove footnote references (superscript numbers)
    text = re.sub(r"\s+\d+(?:(?:,\s*|\s*)\d+)*(?=[\s。，,.;；])", "", text)

    # Remove bullet markers
    text = re.sub(r"^\s*[\*\-]\s+", "", text, flags=re.MULTILINE)

    # Remove numbered list markers
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# TTS synthesis (audio + subtitles in one pass)
# ---------------------------------------------------------------------------

def synthesize(
    text: str,
    voice: str,
    speed: float,
    output_path: Path,
    srt_path: Path | None = None,
) -> None:
    """Synthesize audio via edge-tts, optionally generating SRT subtitles."""
    rate = f"{int((speed - 1) * 100):+d}%"
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    submaker = edge_tts.SubMaker()

    total_chars = len(text)
    written_bytes = 0
    last_pct = -1

    with open(output_path, "wb") as f:
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
                written_bytes += len(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                submaker.feed(chunk)
                offset = chunk.get("offset", 0)
                pct = min(int(offset / total_chars * 100), 99) if total_chars else 0
                if pct > last_pct:
                    last_pct = pct
                    mb = written_bytes / (1024 * 1024)
                    print(f"\r  [{pct:3d}%] {mb:.1f} MB written", end="", flush=True)

    print(f"\r  [100%] {written_bytes / (1024 * 1024):.1f} MB written")

    if srt_path:
        srt_content = submaker.get_srt()
        if srt_content:
            srt_path.write_text(srt_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def generate_video(audio_path: Path, video_path: Path, srt_path: Path | None = None) -> None:
    """Generate video with waveform visualization from audio."""
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found, cannot generate video", file=sys.stderr)
        sys.exit(1)

    filter_complex = (
        "[0:a]showwaves=s=1280x720:mode=cline:rate=30"
        ":colors=0x4a90d9@0.8|0x7ec8e3@0.6:scale=sqrt[waves];"
        "color=c=0x1a1a2e:s=1280x720:r=30[bg];"
        "[bg][waves]overlay=format=auto[v]"
    )

    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "0:a",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-y", str(video_path),
    ]

    print("Generating video (this may take a while)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)

    # Burn in subtitles if available
    if srt_path and srt_path.exists():
        tmp_video = video_path.with_suffix(".tmp.mp4")
        sub_cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"subtitles={srt_path}:force_style='FontSize=24,PrimaryColour=&Hffffff&'",
            "-c:a", "copy", "-y", str(tmp_video),
        ]
        result = subprocess.run(sub_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            tmp_video.replace(video_path)
        else:
            print("Warning: failed to burn subtitles into video", file=sys.stderr)
            tmp_video.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="md2audio",
        description="Convert markdown/text files to audio, with optional subtitles and video.",
    )
    parser.add_argument("input", type=Path, help="Input file (markdown or plain text)")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output audio file (default: ~/w/<stem>.mp3)")

    # TTS options
    tts = parser.add_argument_group("TTS options")
    tts.add_argument("--voice", default=DEFAULT_VOICE, help=f"Voice name (default: {DEFAULT_VOICE})")
    tts.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                     help=f"Speech speed multiplier (default: {DEFAULT_SPEED})")

    # Output options
    out = parser.add_argument_group("Output options")
    out.add_argument("--srt", action="store_true", default=False,
                     help="Also generate SRT subtitles")
    out.add_argument("--video", action="store_true", default=False,
                     help="Also generate MP4 video with waveform visualization")

    # Processing toggles (on by default, use --no-X to disable)
    proc = parser.add_argument_group("Processing (enabled by default)")
    proc.add_argument("--no-strip", action="store_true",
                      help="Don't strip markdown formatting")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    stem = args.input.stem
    output = args.output or Path.home() / "w" / f"{stem}.mp3"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Read and process text
    raw = args.input.read_text(encoding="utf-8")
    text = raw if args.no_strip else strip_markdown(raw)

    print(f"Input:  {args.input} ({len(raw)} chars)")
    print(f"Text:   {len(text)} chars (after processing)")
    print(f"Voice:  {args.voice} @ {args.speed}x")
    print(f"Output: {output}")

    # Generate audio (and subtitles in the same pass)
    srt_path = output.with_suffix(".srt") if args.srt else None
    print("Synthesizing..." + (" (with subtitles)" if srt_path else ""))
    synthesize(text, args.voice, args.speed, output, srt_path)

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Audio: {size_mb:.1f} MB")
    if srt_path and srt_path.exists():
        print(f"Subtitles: {srt_path}")

    # Generate video
    if args.video:
        video_path = output.with_suffix(".mp4")
        generate_video(output, video_path, srt_path if srt_path and srt_path.exists() else None)
        if video_path.exists():
            vid_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"Video: {vid_mb:.1f} MB → {video_path}")

    print("Done!")


if __name__ == "__main__":
    main()
