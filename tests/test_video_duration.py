"""Waveform videos finish with audio, including after subtitle burning."""

import json
import shutil
import subprocess

import pytest

import md2audio


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="FFmpeg integration requires ffmpeg and ffprobe")
@pytest.mark.parametrize("with_subtitles", [False, True])
def test_waveform_video_has_no_silent_tail(tmp_path, monkeypatch, with_subtitles):
    audio, video = tmp_path / "tone.mp3", tmp_path / "wave.mp4"
    real_run = subprocess.run

    def bounded_run(*args, **kwargs):
        kwargs.setdefault("timeout", 30)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(md2audio.subprocess, "run", bounded_run)
    monkeypatch.setattr(md2audio, "_nvenc_available", lambda: False)
    bounded_run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "sine=frequency=440:duration=2.115918", "-c:a", "libmp3lame",
        "-y", str(audio),
    ], check=True)
    subtitle = tmp_path / "captions.srt" if with_subtitles else None
    if subtitle:
        subtitle.write_text("1\n00:00:00,000 --> 00:00:02,000\nPublic test.\n")
    md2audio.generate_video(audio, video, subtitle)
    result = bounded_run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,start_time,duration", "-of", "json", str(video),
    ], capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)
    streams = {stream["codec_type"]: stream for stream in probe["streams"]}
    audio_duration = float(streams["audio"]["duration"])
    assert audio_duration > 2.0
    assert abs(float(streams["video"]["duration"]) - audio_duration) < 0.1
    assert abs(float(probe["format"]["duration"]) - audio_duration) < 0.1
    assert all(abs(float(stream["start_time"])) < 0.05 for stream in streams.values())
