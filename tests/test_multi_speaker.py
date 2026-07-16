"""Offline tests for multi-speaker Markdown dialogue synthesis."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import md2audio


VOICES = [
    {"ShortName": "zh-CN-Male-A", "Gender": "Male", "Locale": "zh-CN", "Status": "GA"},
    {"ShortName": "zh-CN-Male-B", "Gender": "Male", "Locale": "zh-CN", "Status": "Preview"},
    {"ShortName": "zh-TW-Male", "Gender": "Male", "Locale": "zh-TW", "Status": "GA"},
    {"ShortName": "zh-CN-Female", "Gender": "Female", "Locale": "zh-CN", "Status": "GA"},
    {"ShortName": "zh-CN-Deprecated", "Gender": "Male", "Locale": "zh-CN", "Status": "Deprecated"},
]


def test_parse_dialogue_supports_both_colons_and_multiple_paragraphs() -> None:
    turns = md2audio.parse_dialogue(
        "** 主持人 ：** 第一段。\n\n第二段。\n**Guest:** Reply."
    )

    assert turns == [
        md2audio.DialogueTurn("主持人", "第一段。\n\n第二段。"),
        md2audio.DialogueTurn("Guest", "Reply."),
    ]


def test_parse_dialogue_ignores_inline_bold_and_never_returns_labels() -> None:
    turns = md2audio.parse_dialogue("**Host:** Hello **Guest:** still the host's turn.")

    assert turns == [md2audio.DialogueTurn("Host", "Hello **Guest:** still the host's turn.")]


def test_parse_dialogue_rejects_preamble_and_warns_for_empty_turn() -> None:
    with pytest.raises(ValueError, match="before the first"):
        md2audio.parse_dialogue("Title\n**Host:** Hello")

    with pytest.warns(UserWarning, match="empty dialogue turn"):
        turns = md2audio.parse_dialogue("**Host:**\n**Guest:** Hello")
    assert turns == [md2audio.DialogueTurn("Guest", "Hello")]


def test_parse_dialogue_rejects_whitespace_only_speaker_label() -> None:
    with pytest.raises(ValueError, match="empty speaker label"):
        md2audio.parse_dialogue("**    :** Hello")


def test_no_strip_keeps_turn_markdown_but_never_speaks_the_label(tmp_path: Path) -> None:
    input_path = tmp_path / "dialogue.md"
    input_path.write_text("**Host:** Hello **bold** world.", encoding="utf-8")

    _, turns = md2audio._resolve_dialogue(
        SimpleNamespace(input=input_path, output=tmp_path / "dialogue.mp3", no_strip=True)
    )

    assert turns == [md2audio.DialogueTurn("Host", "Hello **bold** world.")]


def test_auto_voice_map_filters_exact_locale_and_assigns_unique_voices() -> None:
    mapping = md2audio.resolve_auto_voice_map(
        ["Host", "Guest", "Host"], VOICES, "male", locale="zh-CN", seed=4
    )

    assert set(mapping) == {"Host", "Guest"}
    assert set(mapping.values()) == {"zh-CN-Male-A", "zh-CN-Male-B"}


def test_auto_voice_map_filters_language_and_is_seeded() -> None:
    first = md2audio.resolve_auto_voice_map(
        ["Host", "Guest"], VOICES, "male", language="zh", seed=17
    )
    second = md2audio.resolve_auto_voice_map(
        ["Host", "Guest"], VOICES, "male", language="zh", seed=17
    )

    assert first == second
    assert set(first.values()).issubset({"zh-CN-Male-A", "zh-CN-Male-B", "zh-TW-Male"})


def test_auto_voice_map_rejects_insufficient_candidates() -> None:
    with pytest.raises(ValueError, match=r"2 speakers, 1 available"):
        md2audio.resolve_auto_voice_map(
            ["Host", "Guest"], VOICES, "female", locale="zh-CN"
        )


def test_load_speaker_config_reads_toml_and_trims_names(tmp_path: Path) -> None:
    config_path = tmp_path / "speakers.toml"
    config_path.write_text(
        '[speakers." 主持人 "]\nvoice = "zh-CN-Male-A"\n\n'
        '[speakers.Guest]\nvoice = "zh-CN-Female"\n',
        encoding="utf-8",
    )

    assert md2audio.load_speaker_config(config_path) == {
        "主持人": "zh-CN-Male-A",
        "Guest": "zh-CN-Female",
    }


def test_config_mapping_reports_missing_extra_unavailable_and_deprecated() -> None:
    with pytest.raises(ValueError, match="Host, Guest"):
        md2audio.resolve_config_voice_map(["Host", "Guest"], {}, VOICES)

    with pytest.warns(UserWarning, match="unused"):
        mapping = md2audio.resolve_config_voice_map(
            ["Host"], {"Host": "zh-CN-Male-A", "Unused": "zh-CN-Female"}, VOICES
        )
    assert mapping == {"Host": "zh-CN-Male-A"}

    with pytest.raises(ValueError, match="unavailable"):
        md2audio.resolve_config_voice_map(["Host"], {"Host": "does-not-exist"}, VOICES)
    with pytest.raises(ValueError, match="deprecated"):
        md2audio.resolve_config_voice_map(["Host"], {"Host": "zh-CN-Deprecated"}, VOICES)


def test_dialogue_synthesis_concatenates_audio_in_turn_order_and_offsets_srt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    segments = iter([
        (b"first", [{"offset": 0, "duration": 20_000_000, "text": "firstwords"}]),
        (b"second", [{"offset": 0, "duration": 10_000_000, "text": "secondword"}]),
    ])
    monkeypatch.setattr(md2audio, "synthesize_dialogue_segment", lambda *_: next(segments))
    monkeypatch.setattr(
        md2audio,
        "_mp3_duration_ticks",
        lambda audio: 30_000_000 if audio == b"first" else 10_000_000,
    )
    output = tmp_path / "dialogue.mp3"

    boundaries = md2audio.synthesize_dialogue(
        [md2audio.DialogueTurn("Host", "first"), md2audio.DialogueTurn("Guest", "second")],
        {"Host": "voice-a", "Guest": "voice-b"},
        1.0,
        output,
        cache_dir=tmp_path / "cache",
    )

    assert output.read_bytes() == b"firstsecond"
    assert boundaries[1]["offset"] == 30_000_000
    assert "00:00:03,000 --> 00:00:04,000" in md2audio.build_quality_srt(boundaries)


def test_dialogue_synthesis_keeps_existing_output_when_a_turn_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "dialogue.mp3"
    output.write_bytes(b"previous-output")
    calls = 0

    def fake_segment(*_args: object) -> tuple[bytes, list[dict]]:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError("network failed")
        return b"partial", [{"offset": 0, "duration": 1, "text": "text"}]

    monkeypatch.setattr(md2audio, "synthesize_dialogue_segment", fake_segment)
    monkeypatch.setattr(md2audio.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match=r"turn 2.*Guest.*voice-b"):
        md2audio.synthesize_dialogue(
            [md2audio.DialogueTurn("Host", "one"), md2audio.DialogueTurn("Guest", "two")],
            {"Host": "voice-a", "Guest": "voice-b"},
            1.0,
            output,
            cache_dir=tmp_path / "cache",
        )
    assert output.read_bytes() == b"previous-output"
    assert not list(tmp_path.glob(".dialogue.*.partial"))
    assert len(list((tmp_path / "cache").glob("*.mp3"))) == 1
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_dialogue_cache_reuses_completed_turns_after_a_failed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "dialogue.mp3"
    output.write_bytes(b"previous-output")
    xdg_cache_home = tmp_path / "xdg-cache"
    cache_dir = xdg_cache_home / "md2audio" / "dialogue" / "turns"
    calls: list[str] = []
    fail_second_turn = True

    def fake_segment(text: str, *_args: object) -> tuple[bytes, list[dict]]:
        calls.append(text)
        if text == "two" and fail_second_turn:
            raise OSError("temporary network failure")
        return text.encode(), [{"offset": 0, "duration": 1, "text": text}]

    turns = [
        md2audio.DialogueTurn("Host", "one"),
        md2audio.DialogueTurn("Guest", "two"),
        md2audio.DialogueTurn("Host", "three"),
    ]
    voices = {"Host": "voice-a", "Guest": "voice-b"}
    monkeypatch.setattr(md2audio, "synthesize_dialogue_segment", fake_segment)
    monkeypatch.setattr(md2audio.time, "sleep", lambda _: None)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache_home))

    with pytest.raises(RuntimeError, match=r"turn 2.*Guest.*voice-b"):
        md2audio.synthesize_dialogue(turns, voices, 1.0, output)

    assert output.read_bytes() == b"previous-output"
    assert calls == ["one", "two", "two", "two", "two", "two"]
    assert len(list(cache_dir.glob("*.mp3"))) == 1

    calls.clear()
    fail_second_turn = False
    md2audio.synthesize_dialogue(turns, voices, 1.0, output)

    assert calls == ["two", "three"]
    assert output.read_bytes() == b"onetwothree"


def test_dialogue_cache_identity_invalidates_only_changed_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "dialogue.mp3"
    cache_dir = tmp_path / "cache"
    calls: list[tuple[str, str]] = []

    def fake_segment(text: str, voice: str, _speed: float) -> tuple[bytes, list[dict]]:
        calls.append((text, voice))
        return f"{voice}:{text}".encode(), [{"offset": 0, "duration": 1, "text": text}]

    monkeypatch.setattr(md2audio, "synthesize_dialogue_segment", fake_segment)
    original_turns = [
        md2audio.DialogueTurn("Host", "one"),
        md2audio.DialogueTurn("Guest", "two"),
    ]
    voices = {"Host": "voice-a", "Guest": "voice-b"}
    md2audio.synthesize_dialogue(original_turns, voices, 1.0, output, cache_dir=cache_dir)

    calls.clear()
    changed_text_turns = [
        md2audio.DialogueTurn("Host", "one updated"),
        md2audio.DialogueTurn("Guest", "two"),
    ]
    md2audio.synthesize_dialogue(changed_text_turns, voices, 1.0, output, cache_dir=cache_dir)
    assert calls == [("one updated", "voice-a")]

    calls.clear()
    changed_voice_map = {"Host": "voice-a", "Guest": "voice-c"}
    md2audio.synthesize_dialogue(
        changed_text_turns, changed_voice_map, 1.0, output, cache_dir=cache_dir,
    )
    assert calls == [("two", "voice-c")]


def test_dialogue_no_cache_always_synthesizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "dialogue.mp3"
    cache_dir = tmp_path / "cache"
    calls: list[str] = []

    def fake_segment(text: str, *_args: object) -> tuple[bytes, list[dict]]:
        calls.append(text)
        return text.encode(), [{"offset": 0, "duration": 1, "text": text}]

    monkeypatch.setattr(md2audio, "synthesize_dialogue_segment", fake_segment)
    turns = [md2audio.DialogueTurn("Host", "one"), md2audio.DialogueTurn("Guest", "two")]
    voices = {"Host": "voice-a", "Guest": "voice-b"}

    md2audio.synthesize_dialogue(turns, voices, 1.0, output, cache_dir=cache_dir, use_cache=False)
    md2audio.synthesize_dialogue(turns, voices, 1.0, output, cache_dir=cache_dir, use_cache=False)

    assert calls == ["one", "two", "one", "two"]
    assert not cache_dir.exists()


def test_dialogue_retries_rate_limits_before_writing_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "dialogue.mp3"
    calls = 0

    class RateLimited(Exception):
        status_code = 429

    def fake_segment(*_args: object) -> tuple[bytes, list[dict]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimited("too many requests")
        return b"recovered", [{"offset": 0, "duration": 1, "text": "recovered"}]

    monkeypatch.setattr(md2audio, "synthesize_dialogue_segment", fake_segment)
    monkeypatch.setattr(md2audio.time, "sleep", lambda _: None)

    md2audio.synthesize_dialogue(
        [md2audio.DialogueTurn("Host", "one")],
        {"Host": "voice-a"},
        1.0,
        output,
        cache_dir=tmp_path / "cache",
    )

    assert calls == 2
    assert output.read_bytes() == b"recovered"


def test_dialogue_async_stream_errors_reach_the_retry_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    class FakeCommunicate:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def stream(self):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                yield {"type": "audio", "data": b"discarded"}
                raise OSError("503 upstream unavailable")
            yield {"type": "audio", "data": b"recovered"}
            yield {"type": "WordBoundary", "offset": 0, "duration": 1, "text": "ok"}

    monkeypatch.setattr(md2audio.edge_tts, "Communicate", FakeCommunicate)
    monkeypatch.setattr(md2audio.time, "sleep", lambda _: None)

    assert md2audio._synthesize_dialogue_turn_with_retries("text", "voice", 1.0) == (
        b"recovered", [{"offset": 0, "duration": 1, "text": "ok"}],
    )
    assert attempts == 2


def test_dialogue_async_stream_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    class HangingCommunicate:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def stream(self):
            await asyncio.Event().wait()
            yield {"type": "audio", "data": b"unreachable"}

    monkeypatch.setattr(md2audio.edge_tts, "Communicate", HangingCommunicate)
    monkeypatch.setattr(md2audio, "_DIALOGUE_TURN_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(TimeoutError):
        md2audio.synthesize_dialogue_segment("text", "voice", 1.0)


def test_dialogue_retry_classifies_aiohttp_and_edge_transient_errors() -> None:
    class ClientPayloadError(Exception):
        pass

    ClientPayloadError.__module__ = "aiohttp.client_exceptions"

    class WebSocketError(Exception):
        pass

    WebSocketError.__module__ = "edge_tts.exceptions"

    assert md2audio._is_retryable_dialogue_error(ClientPayloadError("connection reset"))
    assert md2audio._is_retryable_dialogue_error(WebSocketError("connection reset"))
    assert md2audio._is_retryable_dialogue_error(TimeoutError("timed out"))
    assert not md2audio._is_retryable_dialogue_error(PermissionError("read-only"))


def test_dialogue_cache_manifest_never_pairs_audio_with_other_writer_metadata(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    identity = md2audio._dialogue_turn_identity(md2audio.DialogueTurn("Host", "text"), "voice", 1.0)
    first = (b"first", [{"offset": 0, "duration": 1, "text": "first"}])
    second = (b"second", [{"offset": 0, "duration": 1, "text": "second"}])
    start = threading.Barrier(2)

    def write_cached_turn(audio: bytes, boundaries: list[dict]) -> None:
        start.wait()
        md2audio._write_cached_dialogue_turn(cache_dir, identity, audio, boundaries)

    first_writer = threading.Thread(target=write_cached_turn, args=first)
    second_writer = threading.Thread(target=write_cached_turn, args=second)
    first_writer.start()
    second_writer.start()
    first_writer.join()
    second_writer.join()

    assert md2audio._load_cached_dialogue_turn(cache_dir, identity) in (first, second)


def test_dialogue_cache_preflight_failure_keeps_successful_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    output = tmp_path / "dialogue.mp3"
    original_mkstemp = md2audio.tempfile.mkstemp

    def fail_cache_preflight(*args: object, **kwargs: object) -> tuple[int, str]:
        if kwargs.get("dir") == cache_dir:
            raise OSError("cache unavailable")
        return original_mkstemp(*args, **kwargs)

    monkeypatch.setattr(md2audio.tempfile, "mkstemp", fail_cache_preflight)
    monkeypatch.setattr(
        md2audio,
        "synthesize_dialogue_segment",
        lambda *_: (b"audio", [{"offset": 0, "duration": 1, "text": "text"}]),
    )

    with pytest.warns(UserWarning, match="continuing without cache"):
        md2audio.synthesize_dialogue(
            [md2audio.DialogueTurn("Host", "text")], {"Host": "voice"}, 1.0, output,
            cache_dir=cache_dir,
        )

    assert output.read_bytes() == b"audio"
    assert cache_dir.exists()
    assert not list(cache_dir.iterdir())


def test_dialogue_cache_write_failure_keeps_successful_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "dialogue.mp3"
    writes = 0

    def fail_cache_write(*_args: object) -> None:
        nonlocal writes
        writes += 1
        raise OSError("disk full")

    monkeypatch.setattr(
        md2audio,
        "synthesize_dialogue_segment",
        lambda text, *_: (text.encode(), [{"offset": 0, "duration": 1, "text": text}]),
    )
    monkeypatch.setattr(md2audio, "_write_cached_dialogue_turn", fail_cache_write)

    with pytest.warns(UserWarning, match="continuing without cache"):
        md2audio.synthesize_dialogue(
            [md2audio.DialogueTurn("Host", "one"), md2audio.DialogueTurn("Guest", "two")],
            {"Host": "voice-a", "Guest": "voice-b"}, 1.0, output, cache_dir=tmp_path / "cache",
        )

    assert output.read_bytes() == b"onetwo"
    assert writes == 1


def test_single_speaker_cli_still_uses_existing_synthesis_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.md"
    output_path = tmp_path / "output.mp3"
    input_path.write_text("**Hello**", encoding="utf-8")
    calls: list[tuple[str, str, float, Path]] = []

    def fake_synthesize(text: str, voice: str, speed: float, output: Path) -> list[dict]:
        calls.append((text, voice, speed, output))
        output.write_bytes(b"single-speaker")
        return []

    monkeypatch.setattr(md2audio, "synthesize", fake_synthesize)
    monkeypatch.setattr(
        sys, "argv", ["md2audio", str(input_path), "-o", str(output_path)]
    )

    md2audio.main()

    assert calls == [("Hello", md2audio.DEFAULT_VOICE, 1.0, output_path)]
    assert output_path.read_bytes() == b"single-speaker"


def test_config_mode_synthesizes_when_voice_list_is_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    input_path = tmp_path / "dialogue.md"
    output_path = tmp_path / "dialogue.mp3"
    config_path = tmp_path / "speakers.toml"
    input_path.write_text("**Host:** Hello", encoding="utf-8")
    config_path.write_text('[speakers.Host]\nvoice = "zh-CN-Male-A"\n', encoding="utf-8")
    args = SimpleNamespace(
        multi_speaker=True,
        input=input_path,
        output=output_path,
        no_strip=False,
        speaker_config=config_path,
        speed=1.0,
        srt=False,
        video=False,
        no_cache=False,
    )
    selected_maps: list[dict[str, str]] = []

    def offline_voice_list() -> list[dict]:
        raise OSError("offline")

    def fake_synthesize_dialogue(
        turns: list[md2audio.DialogueTurn], voice_map: dict[str, str], speed: float, output: Path,
        *, use_cache: bool,
    ) -> list[dict]:
        assert turns == [md2audio.DialogueTurn("Host", "Hello")]
        assert speed == 1.0
        selected_maps.append(voice_map)
        assert use_cache is True
        output.write_bytes(b"audio")
        return []

    monkeypatch.setattr(md2audio, "fetch_voices", offline_voice_list)
    monkeypatch.setattr(md2audio, "synthesize_dialogue", fake_synthesize_dialogue)

    with pytest.warns(UserWarning, match="Could not fetch the Edge voice list"):
        md2audio._cmd_synth(args)

    assert selected_maps == [{"Host": "zh-CN-Male-A"}]
    assert output_path.read_bytes() == b"audio"


def test_multi_speaker_synth_removes_stale_srt_without_new_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output_path = tmp_path / "dialogue.mp3"
    stale_srt = output_path.with_suffix(".srt")
    stale_srt.write_text("stale subtitles", encoding="utf-8")
    args = SimpleNamespace(
        multi_speaker=True, speed=1.0, srt=True, video=True, no_cache=False,
    )
    subtitle_inputs: list[Path | None] = []

    monkeypatch.setattr(
        md2audio,
        "_prepare_multi_speaker",
        lambda _: (
            output_path,
            [md2audio.DialogueTurn("Host", "Hello")],
            {"Host": "voice-a"},
        ),
    )

    def fake_synthesize_dialogue(*_args: object, **_kwargs: object) -> list[dict]:
        output_path.write_bytes(b"new audio")
        return []

    def fake_generate_video(_audio: Path, _video: Path, subtitles: Path | None) -> None:
        subtitle_inputs.append(subtitles)

    monkeypatch.setattr(md2audio, "synthesize_dialogue", fake_synthesize_dialogue)
    monkeypatch.setattr(md2audio, "generate_video", fake_generate_video)

    md2audio._cmd_synth(args)

    assert not stale_srt.exists()
    assert subtitle_inputs == [None]


def test_multi_speaker_rejects_explicit_voice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "dialogue.md"
    input_path.write_text("**Host:** Hello", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "md2audio",
            "synth",
            str(input_path),
            "--multi-speaker",
            "--speaker-config",
            str(tmp_path / "speakers.toml"),
            "--voice",
            "zh-CN-YunyangNeural",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        md2audio.main()

    assert exc_info.value.code == 2
    assert "--voice and --multi-speaker are mutually exclusive" in capsys.readouterr().err
