import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import md2audio


def test_synthesis_propagates_stream_error_and_preserves_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "news.mp3"
    output.write_bytes(b"original")
    error = OSError("connection lost after partial audio")
    closed = False

    class FakeCommunicate:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def stream_sync(self) -> None:
            raise AssertionError("The synchronous bridge hides async stream errors")

        async def stream(self) -> AsyncIterator[dict]:
            nonlocal closed
            try:
                yield {"type": "audio", "data": b"partial"}
                raise error
            finally:
                closed = True

    monkeypatch.setattr(md2audio.edge_tts, "Communicate", FakeCommunicate)

    with pytest.raises(OSError) as caught:
        md2audio.synthesize("news", "voice", 1.0, output)

    assert caught.value is error
    assert closed
    assert output.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [output]
    assert "[100%]" not in capsys.readouterr().out


@pytest.mark.parametrize("partial_audio", [False, True])
def test_synthesis_times_out_and_closes_stalled_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], partial_audio: bool,
) -> None:
    output = tmp_path / "news.mp3"
    output.write_bytes(b"original")
    closed = False

    class FakeCommunicate:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def stream_sync(self) -> None:
            raise AssertionError("The synchronous bridge has no stalled-read deadline")

        async def stream(self) -> AsyncIterator[dict]:
            nonlocal closed
            try:
                if partial_audio:
                    yield {"type": "audio", "data": b"partial"}
                await asyncio.Event().wait()
                yield {"type": "audio", "data": b"unreachable"}
            finally:
                closed = True

    monkeypatch.setattr(md2audio.edge_tts, "Communicate", FakeCommunicate)
    monkeypatch.setattr(md2audio, "_SYNTHESIS_READ_TIMEOUT_SECONDS", 0.01, raising=False)

    with pytest.raises(asyncio.TimeoutError):
        md2audio.synthesize("news", "voice", 1.0, output)

    assert closed
    assert output.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [output]
    assert "[100%]" not in capsys.readouterr().out


def test_synthesis_streams_long_audio_with_boundaries_rate_and_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "news.mp3"
    output.write_bytes(b"original")
    calls: list[tuple[str, str, str]] = []
    closed = False
    chunks = [
        {"type": "audio", "data": b"first"},
        {"type": "WordBoundary", "offset": 0, "duration": 1, "text": "hello"},
        {"type": "audio", "data": b"second"},
        {"type": "SentenceBoundary", "offset": 1, "duration": 2, "text": "world"},
        {"type": "audio", "data": b"third"},
        {"type": "SentenceBoundary", "offset": 3, "duration": 1},
    ]

    class FakeCommunicate:
        def __init__(self, text: str, voice: str, *, rate: str) -> None:
            calls.append((text, voice, rate))

        def stream_sync(self) -> None:
            raise AssertionError("Use the async stream so every read has its own deadline")

        async def stream(self) -> AsyncIterator[dict]:
            nonlocal closed
            try:
                for chunk in chunks:
                    await asyncio.sleep(0.02)
                    assert output.read_bytes() == b"original"
                    yield chunk
            finally:
                closed = True

    monkeypatch.setattr(md2audio.edge_tts, "Communicate", FakeCommunicate)
    monkeypatch.setattr(md2audio, "_SYNTHESIS_READ_TIMEOUT_SECONDS", 0.1, raising=False)

    boundaries = md2audio.synthesize("hello world", "voice", 1.25, output)

    assert calls == [("hello world", "voice", "+25%")]
    assert closed
    assert output.read_bytes() == b"firstsecondthird"
    assert list(tmp_path.iterdir()) == [output]
    assert boundaries == [
        {"offset": 0, "duration": 1, "text": "hello"},
        {"offset": 1, "duration": 2, "text": "world"},
        {"offset": 3, "duration": 1, "text": ""},
    ]
    progress = capsys.readouterr().out
    assert "[ 45%]" in progress
    assert "[ 90%]" in progress
    assert progress.count("[100%]") == 1
