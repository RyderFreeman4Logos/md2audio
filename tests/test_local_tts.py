"""Offline TTS contracts: explicit routing, volume, and native process boundaries."""

import argparse
import array
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import md2audio


def test_local_options_default_to_melo_speaker_zero_and_double_volume() -> None:
    parser = argparse.ArgumentParser()
    md2audio._add_common(parser)
    args = parser.parse_args(["--local-tts", "input.md"])

    assert args.local_tts is True
    assert args.local_model is None
    assert args.voice is None
    assert args.volume is None
    assert args.threads is None


@pytest.mark.parametrize("command", ["synth", "bundle"])
@pytest.mark.parametrize("options,message", [(["--threads", "3"], "requires --local-tts"),
                                            (["--local-tts", "--threads", "0"], "positive integer"),
                                            (["--local-tts", "--threads", "-1"], "positive integer"),
                                            (["--local-tts", "--threads", "1.5"], "invalid int")])
def test_threads_cli_rejects_invalid_options_before_io(
    command: str, options: list[str], message: str, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(md2audio, "_resolve", lambda *args: pytest.fail("input IO before validation"))
    monkeypatch.setattr(md2audio, "fetch_voices", lambda *args: pytest.fail("Edge before validation"))
    monkeypatch.setattr(sys, "argv", ["md2audio", command, "missing.md", *options])
    with pytest.raises(SystemExit) as error:
        md2audio.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("volume", ["0", "0.00", "1", "1.50", "2.35", "1000.00"])
def test_local_volume_accepts_nonnegative_values_with_two_decimal_places(volume: str) -> None:
    parser = argparse.ArgumentParser()
    md2audio._add_common(parser)
    args = parser.parse_args(["input.md", "--local-tts", "--volume", volume])

    assert args.volume == volume


@pytest.mark.parametrize("volume", ["-1", "nan", "inf", "2.345", "1e2", ""])
def test_local_volume_rejects_invalid_values_before_io(volume: str) -> None:
    parser = argparse.ArgumentParser()
    md2audio._add_common(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["input.md", "--local-tts", "--volume", volume])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["synth", "bundle"])
@pytest.mark.parametrize("threads", [None, 3])
def test_local_cli_routes_offline_with_default_or_selected_gain(
    command: str, threads: int | None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "input.md"
    output = tmp_path / "audio.mp3"
    model = tmp_path / "model"
    source.write_text("**你好**", encoding="utf-8")
    calls: list[tuple] = []

    def local(text: str, model_dir: Path, speaker: str, speed: float, out: Path,
              volume: str, threads: int | None = None) -> list[dict]:
        calls.append((text, model_dir, speaker, speed, out, volume, threads))
        out.write_bytes(b"offline")
        return [{"text": text, "offset": 0, "duration": 10_000_000}]

    def forbidden(*args, **kwargs):
        pytest.fail("local CLI contacted Edge")

    monkeypatch.setitem(sys.modules, "md2audio_local", SimpleNamespace(synthesize_local=local, LocalTTSError=RuntimeError))
    monkeypatch.setitem(sys.modules, "md2audio_models", SimpleNamespace(ensure_model=lambda name: model, ensure_native=lambda: "/fixture/sherpa"))
    monkeypatch.setattr(md2audio.shutil, "which", lambda name: "/fixture/" + name)
    monkeypatch.setattr(md2audio, "synthesize", forbidden)
    monkeypatch.setattr(md2audio, "fetch_voices", forbidden)
    monkeypatch.setattr(md2audio, "generate_video", lambda *args: None)
    options = [] if command == "synth" else ["--local-model", "aishell3", "--voice", "7", "--volume", "2.35"]
    if threads is not None:
        options += ["--threads", str(threads)]
    monkeypatch.setattr(sys, "argv", ["md2audio", command, str(source), "-o", str(output),
                                    "--local-tts", *options])
    md2audio.main()

    assert calls == [("你好", model, "0" if command == "synth" else "7", 1.0,
                      output, "2.00" if command == "synth" else "2.35", threads)]
    assert output.read_bytes() == b"offline"
    if command == "bundle":
        assert output.with_suffix(".srt").read_text(encoding="utf-8").endswith("你好\n")


@pytest.mark.parametrize("options", [["--volume", "1.50"],
                                    ["--local-tts", "--multi-speaker"],
                                    ["--local-tts", "--voice", "Yunxi"],
                                    ["--local-tts", "--speed", "nan"],
                                    ["--local-tts", "--speed", "0"]])
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
    import md2audio_local
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
if os.environ.get("TTS_TEST_ENV"):
    Path(os.environ["TTS_TEST_ENV"]).write_text(os.environ.get("LD_LIBRARY_PATH", ""))
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
    monkeypatch.setitem(sys.modules, "md2audio_models", SimpleNamespace(ensure_model=lambda name: model, ensure_native=lambda: str(executable)))
    return model, log


def test_local_native_cli_chunks_utf8_applies_gain_once_and_writes_real_mp3(
    native_tts: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, log = native_tts
    source = tmp_path / "input.md"
    output = tmp_path / "result.mp3"
    text = "你好，今天的新闻。" + "长" * 900
    source.write_text(text, encoding="utf-8")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {2, 4, 6})
    monkeypatch.setattr(sys, "argv", ["md2audio", str(source), "-o", str(output),
                                    "--local-tts", "--volume", "2.35", "--srt"])
    md2audio.main()

    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert "".join(call[-1] for call in calls) == text
    assert all(len(call[-1]) <= 400 for call in calls)
    assert all("--sid=0" in call and "--num-threads=3" in call for call in calls)
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


@pytest.mark.parametrize("affinity,cpus,threads,expected", [({2, 4, 6}, 64, None, 3),
                                                        ({2, 4, 6}, 64, 5, 5),
                                                        (None, 7, None, 7),
                                                        (None, None, None, 1),
                                                        (None, 0, None, 1),
                                                        (set(), 7, None, 1)])
def test_local_threads_use_available_cpus_or_explicit_override(
    native_tts: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    affinity: set[int] | None, cpus: int | None, threads: int | None, expected: int,
) -> None:
    from md2audio_local import synthesize_local
    model, log = native_tts

    def available(pid: int) -> set[int]:
        if affinity is None:
            raise OSError("affinity unavailable")
        return affinity

    if affinity is None and cpus is None:
        monkeypatch.delattr(os, "sched_getaffinity")
    else:
        monkeypatch.setattr(os, "sched_getaffinity", available)
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    synthesize_local("你好。", model, "0", 1.0, tmp_path / "result.mp3", threads=threads)
    assert f"--num-threads={expected}" in json.loads(log.read_text(encoding="utf-8"))


@pytest.mark.parametrize("threads", [0, -1, True, 1.5, "3"])
def test_local_invalid_threads_are_rejected_before_setup_or_output(
    threads: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import md2audio_local
    output = tmp_path / "missing-parent" / "audio.mp3"
    monkeypatch.setattr(md2audio_local, "ensure_native", lambda: pytest.fail("native setup before validation"))
    with pytest.raises(md2audio_local.LocalTTSError, match="threads.*positive integer"):
        md2audio_local.synthesize_local("你好。", tmp_path / "missing-model", "0", 1.0,
                                       output, threads=threads)
    assert not output.parent.exists()


def test_local_missing_path_runtime_is_prepared_lazily_and_invoked_directly(
    native_tts: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil
    import md2audio_local
    model, log = native_tts
    runtime = tmp_path / "sherpa-onnx-offline-tts"
    original_which = shutil.which
    calls: list[str] = []

    def prepare() -> str:
        calls.append("native")
        return str(runtime)

    monkeypatch.setattr(shutil, "which", lambda name: None if name == "sherpa-onnx-offline-tts"
                        else original_which(name))
    monkeypatch.setitem(sys.modules, "md2audio_models", SimpleNamespace(
        ensure_model=lambda name: model, ensure_native=prepare,
    ))
    monkeypatch.setattr(md2audio_local, "ensure_native", prepare, raising=False)
    source = tmp_path / "input.md"
    source.write_text("你好。", encoding="utf-8")
    output = tmp_path / "first-use.mp3"
    environment = tmp_path / "native-env.txt"
    monkeypatch.setenv("TTS_TEST_ENV", str(environment))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/original/library-path")
    monkeypatch.setattr(sys, "argv", ["md2audio", str(source), "-o", str(output),
                                    "--local-tts", "--volume", "1.0", "--srt"])
    md2audio.main()
    assert calls == ["native", "native"]
    assert len(log.read_text().splitlines()) == 1
    assert output.stat().st_size > 0
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
                   check=True, capture_output=True)
    assert output.with_suffix(".srt").is_file()
    assert environment.read_text() == str(runtime.parent) + os.pathsep + "/original/library-path"
    assert os.environ["LD_LIBRARY_PATH"] == "/original/library-path"


def test_existing_path_native_keeps_its_original_library_environment(native_tts, tmp_path, monkeypatch) -> None:
    from md2audio_local import synthesize_local
    model, _ = native_tts
    environment = tmp_path / "existing-native-env.txt"
    monkeypatch.setenv("TTS_TEST_ENV", str(environment))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/original/library-path")
    synthesize_local("你好。", model, "0", 1.0, tmp_path / "existing.mp3", "1.00")
    assert environment.read_text() == "/original/library-path"
    assert os.environ["LD_LIBRARY_PATH"] == "/original/library-path"


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
    with pytest.raises(LocalTTSError, match="ffmpeg not found on PATH"):
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


def test_local_nonzero_exit_signals_only_while_leader_identity_is_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import md2audio_local
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    events: list[str] = []

    class ExitedProcess:
        pid = 12345
        stderr = os.fdopen(read_fd, "rb", buffering=0)
        reaped = False

        def poll(self):
            events.append("poll-reap")
            self.reaped = True
            return 3

        def wait(self, timeout=None):
            events.append("wait-reap")
            self.reaped = True
            return 3

    process = ExitedProcess()

    def observe(*args):
        assert not process.reaped
        events.append("observe-unreaped")
        return SimpleNamespace(si_pid=process.pid, si_status=3)

    def signal_group(*args):
        assert not process.reaped, "must never signal a group after reaping its leader"
        events.append("signal-group")

    monkeypatch.setattr(md2audio_local.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(md2audio_local.os, "waitid", observe)
    monkeypatch.setattr(md2audio_local.os, "killpg", signal_group)
    with pytest.raises(md2audio_local.LocalTTSError, match="exited 3"):
        md2audio_local._run_process(["fixture"])
    assert events.index("signal-group") < events.index("wait-reap")
    assert "poll-reap" not in events


def test_local_cleanup_has_its_own_bound_and_reports_unsettled_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import md2audio_local
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    waits: list[float] = []
    signals: list[int] = []

    class StalledProcess:
        pid = 12345
        stderr = os.fdopen(read_fd, "rb", buffering=0)

        def poll(self):
            return None

        def wait(self, timeout=None):
            assert timeout is not None and 0 < timeout <= 5, "cleanup waits must be bounded"
            waits.append(timeout)
            raise subprocess.TimeoutExpired("fixture", timeout)

    process = StalledProcess()
    clock = iter([0.0, 301.0, 301.0, 301.1, 301.2, 306.1])
    monkeypatch.setattr(md2audio_local.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(md2audio_local.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(md2audio_local.os, "waitid", lambda *a: None)
    monkeypatch.setattr(md2audio_local.os, "killpg", lambda pid, sig: signals.append(sig))
    with pytest.raises(md2audio_local.LocalTTSError, match="cleanup.*unsettled"):
        md2audio_local._run_process(["fixture"])
    assert len(signals) == 1 and waits


def test_local_cleanup_retries_second_interrupt_before_mandatory_signal_and_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import md2audio_local
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    events: list[str] = []

    class InterruptedProcess:
        pid = 12345
        stderr = os.fdopen(read_fd, "rb", buffering=0)

        def poll(self):
            return None

        def wait(self, timeout=None):
            assert timeout is not None and timeout > 0
            if "wait-interrupt" not in events:
                events.append("wait-interrupt")
                raise KeyboardInterrupt
            events.append("reaped")
            return -9

    process = InterruptedProcess()
    clock = iter([0.0, 301.0, 301.0, 301.1, 301.2, 301.3, 301.4, 301.5])

    def signal_group(*args):
        if "signal-interrupt" not in events:
            events.append("signal-interrupt")
            raise KeyboardInterrupt
        events.append("signal-sent")

    monkeypatch.setattr(md2audio_local.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(md2audio_local.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(md2audio_local.os, "waitid", lambda *a: None)
    monkeypatch.setattr(md2audio_local.os, "killpg", signal_group)
    try:
        with pytest.raises(md2audio_local.LocalTTSError, match="timed out"):
            md2audio_local._run_process(["fixture"])
    except KeyboardInterrupt:
        pytest.fail("cleanup allowed a second interruption to escape before termination")
    assert events == ["signal-interrupt", "signal-sent", "wait-interrupt", "reaped"]


def test_local_descendant_held_stderr_is_killed_before_reaping_the_exited_leader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import md2audio_local
    descendant_pid = tmp_path / "descendant.pid"
    roots: list[subprocess.Popen] = []
    signals: list[int] = []
    popen = subprocess.Popen
    killpg = os.killpg
    waitid = os.waitid

    def spawn(*args, **kwargs):
        process = popen(*args, **kwargs)
        roots.append(process)
        return process

    def signal_owned_group(pid: int, sig: int):
        observed = waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        assert observed is not None and observed.si_pid == pid and observed.si_status == 3
        signals.append(pid)
        return killpg(pid, sig)

    monkeypatch.setattr(md2audio_local.subprocess, "Popen", spawn)
    monkeypatch.setattr(md2audio_local.os, "killpg", signal_owned_group)
    monkeypatch.setattr(md2audio_local, "_PROCESS_TIMEOUT_SECONDS", 0.3)
    script = """import subprocess, sys
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path(sys.argv[1]).write_text(str(child.pid))
sys.exit(3)
"""
    try:
        with pytest.raises(md2audio_local.LocalTTSError, match="timed out"):
            md2audio_local._run_process([sys.executable, "-c", script, str(descendant_pid)])
        assert roots[0].returncode == 3 and signals == [roots[0].pid]
        with pytest.raises(ChildProcessError):
            os.waitpid(roots[0].pid, os.WNOHANG)
        child = int(descendant_pid.read_text())
        state = Path(f"/proc/{child}/stat")
        deadline = time.monotonic() + 2
        while state.exists() and state.read_text().split(")", 1)[1].split()[0] != "Z":
            assert time.monotonic() < deadline, "signaled descendant did not terminate"
            threading.Event().wait(0.01)
        assert not state.exists() or state.read_text().split(")", 1)[1].split()[0] == "Z"
    finally:
        if roots and roots[0].returncode is None:
            killpg(roots[0].pid, 9)
            roots[0].wait(timeout=5)


def test_local_cleanup_blocks_sigint_and_restores_the_callers_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import md2audio_local
    masks: list[tuple] = []
    old_mask = {md2audio_local.signal.SIGTERM}

    def mask(operation, signals):
        masks.append((operation, signals))
        return old_mask

    def wait(timeout):
        assert masks == [(md2audio_local.signal.SIG_BLOCK, {md2audio_local.signal.SIGINT})]
        return 0

    process = SimpleNamespace(pid=12345, wait=wait)
    monkeypatch.setattr(md2audio_local.signal, "pthread_sigmask", mask)
    monkeypatch.setattr(md2audio_local.os, "waitid", lambda *args: None)
    monkeypatch.setattr(md2audio_local.os, "killpg", lambda *args: None)
    assert md2audio_local._terminate_and_reap(process) == 0
    assert masks[-1] == (md2audio_local.signal.SIG_SETMASK, old_mask)


def test_local_paragraph_subtitles_keep_all_text_in_valid_cues(
    native_tts: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, log = native_tts
    source = tmp_path / "paragraphs.md"
    output = tmp_path / "paragraphs.mp3"
    text = "第一段。\n\n第二段。\n\n第三段。"
    source.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["md2audio", str(source), "--local-tts",
                                    "-o", str(output), "--srt"])
    md2audio.main()
    cues = re.split(r"\n\s*\n", output.with_suffix(".srt").read_text().strip())
    bodies: list[str] = []
    for index, cue in enumerate(cues, 1):
        lines = cue.splitlines()
        assert lines[0] == str(index)
        assert re.fullmatch(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", lines[1])
        bodies.extend(lines[2:])
    assert "".join("".join(bodies).split()) == "".join(text.split())
    assert json.loads(log.read_text())[-1] == text


def test_local_nul_text_is_rejected_before_native_invocation(
    native_tts: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import md2audio_local
    model, log = native_tts
    output = tmp_path / "audio.mp3"
    output.write_bytes(b"original")

    def forbidden(*args):
        pytest.fail("NUL input reached a native process")

    monkeypatch.setattr(md2audio_local, "_run_process", forbidden)
    with pytest.raises(md2audio_local.LocalTTSError, match="NUL"):
        md2audio_local.synthesize_local("hello\x00world", model, "0", 1.0, output)
    assert output.read_bytes() == b"original" and not log.exists()


@pytest.mark.parametrize("phase", ["constructor", "handoff", "cleanup_entry"])
def test_local_real_sigint_protects_spawn_handoff_and_cleanup_entry(phase: str) -> None:
    """Send real SIGINT in a private probe; do not interrupt pytest or leak its children."""
    script = '''
import json, os, signal, subprocess, sys, time
import md2audio_local
phase = sys.argv[1]
original_popen = subprocess.Popen
original_cleanup = md2audio_local._terminate_and_reap
old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
owned = []
class ConstructorInterrupt(original_popen):
    def _execute_child(self, *args, **kwargs):
        super()._execute_child(*args, **kwargs)
        owned.append(self)
        os.kill(os.getpid(), signal.SIGINT)
def handoff(*args, **kwargs):
    child = original_popen(*args, **kwargs)
    owned.append(child)
    if phase == "handoff":
        os.kill(os.getpid(), signal.SIGINT)
    return child
def cleanup(child):
    os.kill(os.getpid(), signal.SIGINT)
    return original_cleanup(child)
md2audio_local.subprocess.Popen = ConstructorInterrupt if phase == "constructor" else handoff
if phase == "cleanup_entry":
    md2audio_local._PROCESS_TIMEOUT_SECONDS = 0.15
    md2audio_local._terminate_and_reap = cleanup
started = time.monotonic()
interrupted = False
try:
    try:
        md2audio_local._run_process([sys.executable, "-B", "-c", "import time; time.sleep(30)"])
    except KeyboardInterrupt:
        interrupted = True
    result = {"interrupted": interrupted, "reaped_before_probe_cleanup": owned[0].returncode is not None,
              "stderr_closed": owned[0].stderr.closed, "elapsed": time.monotonic() - started,
              "mask_restored": signal.pthread_sigmask(signal.SIG_BLOCK, set()) == old_mask}
finally:
    for child in owned:
        if child.returncode is None:
            os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
        if child.stderr is not None:
            child.stderr.close()
print(json.dumps(result))
'''
    probe = subprocess.run([sys.executable, "-B", "-c", script, phase], capture_output=True,
                           text=True, timeout=8, check=True)
    result = json.loads(probe.stdout)
    assert result["interrupted"] and result["reaped_before_probe_cleanup"]
    assert result["stderr_closed"] and result["mask_restored"]
    assert result["elapsed"] < 2


@pytest.mark.parametrize("mode", ["custom_handler", "caller_mask"])
def test_local_real_sigint_preserves_the_callers_handler_and_mask(mode: str) -> None:
    script = '''
import json, os, signal, subprocess, sys
import md2audio_local
mode = sys.argv[1]
handled = []
if mode == "custom_handler":
    signal.signal(signal.SIGINT, lambda signum, frame: handled.append(signum))
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
if mode == "caller_mask":
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
old_handler = signal.getsignal(signal.SIGINT)
original_popen = subprocess.Popen
owned = []
def spawn(*args, **kwargs):
    child = original_popen(*args, **kwargs)
    owned.append(child)
    os.kill(os.getpid(), signal.SIGINT)
    return child
md2audio_local.subprocess.Popen = spawn
try:
    md2audio_local._run_process([sys.executable, "-B", "-c", "pass"])
    result = {"handler_unchanged": signal.getsignal(signal.SIGINT) is old_handler,
              "mask_unchanged": signal.pthread_sigmask(signal.SIG_BLOCK, set()) == old_mask,
              "pending": signal.SIGINT in signal.sigpending(), "handled": handled,
              "reaped": owned[0].returncode == 0, "stderr_closed": owned[0].stderr.closed}
finally:
    for child in owned:
        if child.returncode is None:
            os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
        child.stderr.close()
print(json.dumps(result))
'''
    probe = subprocess.run([sys.executable, "-B", "-c", script, mode], capture_output=True,
                           text=True, timeout=8, check=True)
    result = json.loads(probe.stdout)
    assert result["handler_unchanged"] and result["mask_unchanged"]
    assert result["reaped"] and result["stderr_closed"]
    assert result["pending"] == (mode == "caller_mask")
    assert len(result["handled"]) == (1 if mode == "custom_handler" else 0)


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
