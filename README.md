# md2audio

`md2audio` converts Markdown or plain text into an MP3 with optional SRT
subtitles and waveform video. Edge TTS is the default; offline VITS synthesis
through sherpa-onnx is available only when explicitly selected.

```bash
uv sync --extra dev
uv run md2audio input.md -o input.mp3
```

The default single-speaker workflow remains unchanged:

```bash
md2audio synth input.md --voice zh-CN-YunyangNeural --srt --video
md2audio bundle input.md
```

## Offline TTS on Termux

Add the plain `--local-tts` flag to use free offline TTS:

```bash
md2audio input.txt --local-tts -o output.mp3
```

On the first local synthesis, md2audio downloads the default Melo Chinese-English
model (about 167 MB) into `$XDG_DATA_HOME/md2audio/models/vits-melo-tts-zh_en`.
If `XDG_DATA_HOME` is unset, empty, or relative, it uses
`~/.local/share/md2audio/models/` instead. Installation, imports, `--help`, and
Edge synthesis do not download models or touch this data directory. Later local
runs verify and reuse the cached model without network access. Failed downloads
leave the requested MP3 intact and can be retried with the same command.

Install `ffmpeg` and `libc++` in Termux, and put the native
`sherpa-onnx-offline-tts` executable on `PATH`. The
[sherpa-onnx v1.13.8 release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.8)
provides `sherpa-onnx-v1.13.8-android-aarch64-termux-static.tar.bz2` for native
ARM64 Termux; Linux ARM64 binaries do not run directly on Android. These native
tools must be installed before first use; md2audio downloads models only.

The default [Melo Chinese-English VITS model](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html)
has one voice. Omit `--voice` to retain speaker 0. Local volume defaults to
`2.00`, doubling sample amplitude once during MP3 encoding. Override it with
`--volume`, using a nonnegative decimal with at most two decimal places:

```bash
md2audio input.txt --local-tts --volume 1.50 -o output.mp3
```

For more voices, select the free AISHELL3 model with `--local-model aishell3`.
Its first use downloads about 32 MB into the same XDG models directory, under
`vits-icefall-zh-aishell3`. It has 174 voices, selected by numeric `--voice` IDs
0–173; speaker 0 remains the default. This model outputs 8 kHz audio, compared
to Melo's 44.1 kHz output.

```bash
md2audio input.txt --local-tts --local-model aishell3 --voice 10 \
  --volume 2.35 --speed 1.10 --srt -o another-voice.mp3
```

`--local-model` and `--volume` require `--local-tts`. Without `--local-tts`,
synthesis uses Edge. Local synthesis never contacts Edge or falls back to it
on failure. Large volume gains can clip. Downloaded archives and cached model
files are checked against pinned sizes and SHA256 hashes; if existing cached
data is corrupt, md2audio reports its path and preserves it for manual recovery.

Local mode supports `synth`, `bundle`, subtitles, and video, but currently
rejects `--multi-speaker`. It groups text at sentence breaks into chunks of at
most 400 characters, gives each native process 300 seconds plus up to 5 seconds
for cleanup (reporting unsettled cleanup on failure), and replaces
the requested MP3 only after every chunk and conversion succeeds. The CLI
reloads its model per chunk, adding startup cost on long documents. Subtitles
use measured WAV durations at chunk boundaries and approximate timing within
each chunk; the model does not provide word timestamps.

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

### Resume cache

Multi-speaker synthesis caches each successful turn's MP3, boundaries, and
identity metadata under `$XDG_CACHE_HOME/md2audio/dialogue/turns/` (or
`~/.cache/md2audio/dialogue/turns/` when `XDG_CACHE_HOME` is unset). Cache keys
include the processed turn text, speaker, voice, and speed, so a rerun only
requests missing or changed turns. Transient Edge and network failures retry a
bounded number of times; after a final failure the completed turn cache remains
while the requested MP3 is left unchanged.

Use `--no-cache` with `--multi-speaker` to force every turn to be synthesized
without reading or writing the cache. For long transcripts such as a 538-turn
Leopold dialogue, keep the default cache enabled: rerun the same command after
an interruption to resume from completed turns instead of starting over.

Multi-speaker MP3 output is an atomic binary concatenation of the MP3 frames
returned by Edge TTS. This assumes Edge returns the same MP3 encoding for each
request, and adds no `ffmpeg` dependency for audio-only output. Subtitle
boundaries are shifted by the preceding turns' actual boundary duration; when
a turn has no boundaries, `md2audio` falls back to its MP3 frame duration.

## Verification

Tests mock Edge calls and exercise local synthesis with a deterministic native
CLI fixture plus the real FFmpeg executable. They do not use the network or
require a downloaded ONNX model; `ffmpeg` must be installed:

```bash
just check
```

Optional live smoke test (save the dialogue example above as `transcript.md`; this
requires network access to Edge TTS):

```bash
md2audio synth transcript.md -o ~/tmp/dialogue.mp3 \
  --multi-speaker --speaker-gender male --speaker-language en --seed 42
```
