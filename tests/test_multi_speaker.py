"""Offline tests for multi-speaker Markdown dialogue synthesis."""

from __future__ import annotations

import sys
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
    output = tmp_path / "dialogue.mp3"

    boundaries = md2audio.synthesize_dialogue(
        [md2audio.DialogueTurn("Host", "first"), md2audio.DialogueTurn("Guest", "second")],
        {"Host": "voice-a", "Guest": "voice-b"},
        1.0,
        output,
    )

    assert output.read_bytes() == b"firstsecond"
    assert boundaries[1]["offset"] == 20_000_000
    assert "00:00:02,000 --> 00:00:03,000" in md2audio.build_quality_srt(boundaries)


def test_dialogue_synthesis_keeps_existing_output_when_a_turn_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    output = tmp_path / "dialogue.mp3"
    output.write_bytes(b"previous-output")
    calls = 0

    def fake_segment(*_args: object) -> tuple[bytes, list[dict]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("network failed")
        return b"partial", [{"offset": 0, "duration": 1, "text": "text"}]

    monkeypatch.setattr(md2audio, "synthesize_dialogue_segment", fake_segment)

    with pytest.raises(RuntimeError, match=r"turn 2.*Guest.*voice-b"):
        md2audio.synthesize_dialogue(
            [md2audio.DialogueTurn("Host", "one"), md2audio.DialogueTurn("Guest", "two")],
            {"Host": "voice-a", "Guest": "voice-b"},
            1.0,
            output,
        )
    assert output.read_bytes() == b"previous-output"
    assert not list(tmp_path.glob(".dialogue.*.partial"))


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
