"""Offline TTS contracts: explicit routing, volume, and native process boundaries."""

import argparse
import array
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import md2audio


def test_local_options_default_to_melo_speaker_zero_and_double_volume() -> None:
    parser = argparse.ArgumentParser()
    md2audio._add_common(parser)
    args = parser.parse_args(["input.md", "--local-tts", "models/melo"])

    assert args.local_tts == Path("models/melo")
    assert args.voice is None
    assert args.volume is None


@pytest.mark.parametrize("volume", ["0", "0.00", "1", "1.50", "2.35", "1000.00"])
def test_local_volume_accepts_nonnegative_values_with_two_decimal_places(volume: str) -> None:
    parser = argparse.ArgumentParser()
    md2audio._add_common(parser)
    args = parser.parse_args(["input.md", "--local-tts", "models/melo", "--volume", volume])

    assert args.volume == volume


@pytest.mark.parametrize("volume", ["-1", "nan", "inf", "2.345", "1e2", ""])
def test_local_volume_rejects_invalid_values_before_io(volume: str) -> None:
    parser = argparse.ArgumentParser()
    md2audio._add_common(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["input.md", "--local-tts", "models/melo", "--volume", volume])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["synth", "bundle"])
def test_local_cli_routes_offline_with_default_or_selected_gain(
    command: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "input.md"
    output = tmp_path / "audio.mp3"
    model = tmp_path / "model"
    source.write_text("**你好**", encoding="utf-8")
    calls: list[tuple] = []

    def local(text: str, model_dir: Path, speaker: str, speed: float, out: Path, volume: str) -> list[dict]:
        calls.append((text, model_dir, speaker, speed, out, volume))
        out.write_bytes(b"offline")
        return [{"text": text, "offset": 0, "duration": 10_000_000}]

    def forbidden(*args, **kwargs):
        pytest.fail("local CLI contacted Edge")

    monkeypatch.setitem(sys.modules, "md2audio_local", SimpleNamespace(synthesize_local=local))
    monkeypatch.setattr(md2audio, "synthesize", forbidden)
    monkeypatch.setattr(md2audio, "fetch_voices", forbidden)
    monkeypatch.setattr(md2audio, "generate_video", lambda *args: None)
    options = [] if command == "synth" else ["--voice", "7", "--volume", "2.35"]
    monkeypatch.setattr(sys, "argv", ["md2audio", command, str(source), "-o", str(output),
                                    "--local-tts", str(model), *options])
    md2audio.main()

    assert calls == [("你好", model, "0" if command == "synth" else "7", 1.0,
                      output, "2.00" if command == "synth" else "2.35")]
    assert output.read_bytes() == b"offline"
    if command == "bundle":
        assert output.with_suffix(".srt").read_text(encoding="utf-8").endswith("你好\n")


@pytest.mark.parametrize("options", [["--volume", "1.50"],
                                    ["--local-tts", "melo", "--multi-speaker"],
                                    ["--local-tts", "melo", "--voice", "Yunxi"],
                                    ["--local-tts", "melo", "--speed", "nan"],
                                    ["--local-tts", "melo", "--speed", "0"]])
def test_local_cli_rejects_incompatible_selectors_before_input_io(
    options: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["md2audio", "missing.md", *options])
    with pytest.raises(SystemExit) as error:
        md2audio.main()
    assert error.value.code == 2


@pytest.fixture
def native_tts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Exercise the real executable boundary with deterministic PCM, without ONNX."""
    model = tmp_path / "renamed model"
    model.mkdir()
    for name in ("model.onnx", "lexicon.txt", "tokens.txt", "date.fst", "number.fst"):
        (model / name).write_text("fixture", encoding="utf-8")
    log = tmp_path / "commands.jsonl"
    executable = tmp_path / "sherpa-onnx-offline-tts"
    executable.write_text("#!" + sys.executable + "\n" + '''
import json, math, os, struct, sys, wave
from pathlib import Path
with open(os.environ["TTS_TEST_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(sys.argv[1:], ensure_ascii=False) + "\\n")
if os.environ.get("TTS_TEST_FAIL"):
    print("native fixture failure", file=sys.stderr)
    sys.exit(3)
if os.environ.get("TTS_TEST_STALL"):
    import time
    Path(os.environ["TTS_TEST_PID"]).write_text(str(os.getpid()))
    time.sleep(30)
options = sys.argv[1:sys.argv.index("--")] if "--" in sys.argv else sys.argv[1:]
output = next(a.split("=", 1)[1] for a in reversed(options) if a.startswith("--output-filename="))
with wave.open(output, "wb") as wav:
    wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    wav.writeframes(b"".join(struct.pack("<h", int(1000 * math.sin(i * math.tau * 400 / 16000)))
                             for i in range(16000)))
if os.environ.get("TTS_TEST_TRUNCATE"):
    with open(output, "r+b") as wav:
        wav.truncate(100)
''', encoding="utf-8")
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TTS_TEST_LOG", str(log))
    return model, log


def test_local_native_cli_chunks_utf8_applies_gain_once_and_writes_real_mp3(
    native_tts: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, log = native_tts
    source = tmp_path / "input.md"
    output = tmp_path / "result.mp3"
    text = "你好，今天的新闻。" + "长" * 900
    source.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["md2audio", str(source), "-o", str(output),
                                    "--local-tts", str(model), "--volume", "2.35", "--srt"])
    md2audio.main()

    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert "".join(call[-1] for call in calls) == text
    assert all(len(call[-1]) <= 400 for call in calls)
    assert all("--sid=0" in call and "--num-threads=2" in call for call in calls)
    assert all(f"--vits-model={model / 'model.onnx'}" in call for call in calls)
    assert all(f"--tts-rule-fsts={model / 'date.fst'},{model / 'number.fst'}" in call for call in calls)
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "s16le", "-"],
                             check=True, capture_output=True).stdout
    pcm = array.array("h", decoded)
    rms = (sum(sample * sample for sample in pcm) / len(pcm)) ** 0.5
    assert 1550 < rms < 1750  # One 2.35 gain; fixture baseline RMS is about 707.
    assert output.with_suffix(".srt").read_text(encoding="utf-8").startswith(
        "1\n00:00:00,000 --> "
    )
    assert not list(tmp_path.glob(".result.*"))


def test_local_speaker_ids_use_sidecar_order_and_reject_silent_fallback(
    native_tts: tuple[Path, Path], tmp_path: Path,
) -> None:
    from md2audio_local import LocalTTSError, synthesize_local
    model, log = native_tts
    output = tmp_path / "audio.mp3"
    output.write_bytes(b"original")
    with pytest.raises(LocalTTSError, match="Speaker ID 1 is unavailable"):
        synthesize_local("你好。", model, "1", 1.0, output)
    assert not log.exists()
    assert output.read_bytes() == b"original"

    (model / "speakers.txt").write_text("SSB0005\nSSB0009\nSSB0011\n", encoding="utf-8")
    boundaries = synthesize_local("你好。", model, "2", 1.25, output)
    call = json.loads(log.read_text(encoding="utf-8"))
    assert "--sid=2" in call and "--speed=1.25" in call
    assert boundaries == [{"text": "你好。", "offset": 0, "duration": 10_000_000}]
    before = log.read_bytes()
    with pytest.raises(LocalTTSError, match="Speaker ID 3 is unavailable"):
        synthesize_local("你好。", model, "3", 1.0, output)
    assert log.read_bytes() == before


@pytest.mark.parametrize("speaker,speed,volume", [("-1", 1.0, "2.00"),
                                                 ("garbage", 1.0, "2.00"),
                                                 ("0", float("nan"), "2.00"),
                                                 ("0", 0.0, "2.00"),
                                                 ("0", 1.0, "inf"),
                                                 ("0", 1.0, "2.345")])
def test_local_public_api_validates_before_native_invocation(
    native_tts: tuple[Path, Path], tmp_path: Path, speaker: str, speed: float, volume: str,
) -> None:
    from md2audio_local import LocalTTSError, synthesize_local
    model, log = native_tts
    output = tmp_path / "audio.mp3"
    output.write_bytes(b"original")
    with pytest.raises(LocalTTSError):
        synthesize_local("你好。", model, speaker, speed, output, volume)
    assert output.read_bytes() == b"original"
    assert not log.exists()


def test_local_failure_preserves_output_and_cleans_temporary_audio(
    native_tts: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md2audio_local import LocalTTSError, synthesize_local
    model, _ = native_tts
    output = tmp_path / "audio.mp3"
    output.write_bytes(b"original")
    monkeypatch.setenv("TTS_TEST_FAIL", "1")
    with pytest.raises(LocalTTSError, match="native fixture failure"):
        synthesize_local("你好。", model, "0", 1.0, output)
    assert output.read_bytes() == b"original"
    assert not list(tmp_path.glob(".audio.*"))


def test_local_missing_model_or_dependency_is_actionable_and_keeps_output(
    native_tts: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md2audio_local import LocalTTSError, synthesize_local
    model, log = native_tts
    output = tmp_path / "audio.mp3"
    output.write_bytes(b"original")
    tokens = model / "tokens.txt"
    tokens.unlink()
    with pytest.raises(LocalTTSError, match="Missing VITS model file: .*tokens.txt"):
        synthesize_local("你好。", model, "0", 1.0, output)
    tokens.write_text("fixture", encoding="utf-8")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(LocalTTSError, match="sherpa-onnx-offline-tts not found on PATH"):
        synthesize_local("你好。", model, "0", 1.0, output)
    assert not log.exists()
    assert output.read_bytes() == b"original"


def test_local_zero_gain_mutes_without_overriding_the_default(
    native_tts: tuple[Path, Path], tmp_path: Path,
) -> None:
    from md2audio_local import synthesize_local
    model, _ = native_tts
    output = tmp_path / "muted.mp3"
    synthesize_local("你好。", model, "0", 1.0, output, "0.00")
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "s16le", "-"],
                             check=True, capture_output=True).stdout
    assert decoded and not any(array.array("h", decoded))


def test_local_rejects_truncated_native_audio_before_replacing_output(
    native_tts: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from md2audio_local import LocalTTSError, synthesize_local
    model, _ = native_tts
    output = tmp_path / "audio.mp3"
    output.write_bytes(b"original")
    monkeypatch.setenv("TTS_TEST_TRUNCATE", "1")
    with pytest.raises(LocalTTSError, match="truncated"):
        synthesize_local("你好。", model, "0", 1.0, output)
    assert output.read_bytes() == b"original"
    assert not list(tmp_path.glob(".audio.*"))


def test_local_text_that_looks_like_flags_stays_literal(
    native_tts: tuple[Path, Path], tmp_path: Path,
) -> None:
    from md2audio_local import synthesize_local
    model, log = native_tts
    output = tmp_path / "audio.mp3"
    escape = tmp_path / "escaped"
    text = f"--output-filename={escape}"
    boundaries = synthesize_local(text, model, "0", 1.0, output)
    call = json.loads(log.read_text(encoding="utf-8"))
    assert call[-2:] == ["--", text]
    assert boundaries[0]["text"] == text
    assert output.exists() and not escape.exists()


def test_local_timeout_kills_reaps_and_preserves_output(
    native_tts: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import md2audio_local
    model, _ = native_tts
    output = tmp_path / "audio.mp3"
    pid_file = tmp_path / "pid"
    output.write_bytes(b"original")
    monkeypatch.setenv("TTS_TEST_STALL", "1")
    monkeypatch.setenv("TTS_TEST_PID", str(pid_file))
    monkeypatch.setattr(md2audio_local, "_PROCESS_TIMEOUT_SECONDS", 0.15)
    with pytest.raises(md2audio_local.LocalTTSError, match="timed out"):
        md2audio_local.synthesize_local("你好。", model, "0", 1.0, output)
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert output.read_bytes() == b"original"
    assert not list(tmp_path.glob(".audio.*"))
