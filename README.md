# md2audio

`md2audio` converts Markdown or plain text into an MP3 with optional SRT
subtitles and waveform video, powered by Edge TTS.

```bash
uv sync --extra dev
uv run md2audio input.md -o input.mp3
```

The default single-speaker workflow remains unchanged:

```bash
md2audio synth input.md --voice zh-CN-YunyangNeural --srt --video
md2audio bundle input.md
```

## Multi-speaker dialogue

Add `--multi-speaker` to parse bold speaker labels that begin a line. Labels
are metadata and are never spoken, including when `--no-strip` is set. A turn
continues through subsequent paragraphs until the next label.

```markdown
**主持人：** 欢迎来到节目。

这一段仍然属于主持人。

**Leopold:** Thanks for having me.
```

### Automatic voice assignment

Automatic mode fetches Edge's voice list once, selects non-deprecated voices
of the requested gender, and assigns one distinct voice to each speaker. The
same seed, speaker order, and available voice list produce the same mapping.

```bash
md2audio synth transcript.md -o transcript.mp3 \
  --multi-speaker \
  --speaker-gender male \
  --speaker-locale zh-CN \
  --seed 42

md2audio bundle transcript.md -o transcript.mp3 \
  --multi-speaker \
  --speaker-gender female \
  --speaker-language zh
```

Use exactly one of `--speaker-locale` (for example, `zh-CN`) or
`--speaker-language` (for example, `zh`) with `--speaker-gender`.

### TOML speaker configuration

For an exact per-speaker mapping, pass a TOML file with `--speaker-config`.
This mode can mix languages, locales, and genders. It is mutually exclusive
with the automatic selectors.

```toml
[speakers."主持人"]
voice = "zh-CN-YunxiNeural"

[speakers.Leopold]
voice = "zh-CN-YunyangNeural"
```

```bash
md2audio synth transcript.md -o transcript.mp3 \
  --multi-speaker \
  --speaker-config speakers.toml \
  --srt --video
```

Quoted TOML keys support Unicode names and spaces. Every detected speaker must
be present; unused config entries cause a warning. The current Edge voice list
is checked once for unavailable or deprecated configured voices.

Multi-speaker MP3 output is an atomic binary concatenation of the MP3 frames
returned by Edge TTS. This assumes Edge returns the same MP3 encoding for each
request, and adds no `ffmpeg` dependency for audio-only output. Subtitle
boundaries are shifted by the preceding turns' actual boundary duration; when
a turn has no boundaries, `md2audio` falls back to its MP3 frame duration.

## Verification

Unit tests mock all Edge voice and synthesis calls, so they do not use the
network:

```bash
.venv/bin/python -m pytest -q
```

Optional live smoke test (save the dialogue example above as `transcript.md`; this
requires network access to Edge TTS):

```bash
md2audio synth transcript.md -o /tmp/dialogue.mp3 \
  --multi-speaker --speaker-gender male --speaker-language en --seed 42
```
