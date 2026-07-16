"""Parsing and voice selection for multi-speaker Markdown dialogue."""

import asyncio
from dataclasses import dataclass
import inspect
import random
import re
import warnings
from pathlib import Path
from typing import Any

import edge_tts

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


_DIALOGUE_LABEL_RE = re.compile(
    r"^[ \t]*\*\*(?P<speaker>[^*\r\n:：]+?)[ \t]*[:：][ \t]*\*\*[ \t]*(?P<text>.*)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class DialogueTurn:
    """One speaker's contiguous portion of a dialogue transcript."""

    speaker: str
    text: str


def parse_dialogue(raw_markdown: str) -> list[DialogueTurn]:
    """Parse line-start bold speaker labels from a Markdown transcript.

    Labels are metadata, so they are removed before individual turn bodies are
    optionally passed through the caller's Markdown processor.

    Raises:
        ValueError: The document has no labels or contains a nonblank preamble.
    """
    matches = list(_DIALOGUE_LABEL_RE.finditer(raw_markdown))
    if not matches:
        raise ValueError(
            "Multi-speaker input needs line-start labels such as '**Speaker:** text'."
        )
    if raw_markdown[:matches[0].start()].strip():
        raise ValueError("Multi-speaker input cannot contain content before the first speaker label.")

    turns: list[DialogueTurn] = []
    for index, match in enumerate(matches, 1):
        speaker = match.group("speaker").strip()
        body_end = matches[index].start() if index < len(matches) else len(raw_markdown)
        body = (match.group("text") + raw_markdown[match.end():body_end]).strip()
        if not body:
            warnings.warn(
                f"Ignoring empty dialogue turn for speaker {speaker!r}.",
                UserWarning,
                stacklevel=2,
            )
            continue
        turns.append(DialogueTurn(speaker=speaker, text=body))

    if not turns:
        raise ValueError("Multi-speaker input contains no nonempty dialogue turns.")
    return turns


def load_speaker_config(path: Path) -> dict[str, str]:
    """Load the strict TOML speaker-to-voice mapping at *path*.

    The accepted schema is ``[speakers.<name>]`` with one required ``voice``
    string per speaker. Speaker names are trimmed to match parsed labels.
    """
    try:
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except OSError as exc:
        raise ValueError(f"Could not read speaker config {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML speaker config {path}: {exc}") from exc

    if set(document) != {"speakers"} or not isinstance(document.get("speakers"), dict):
        raise ValueError("Speaker config must contain only a [speakers] table.")

    result: dict[str, str] = {}
    for raw_name, entry in document["speakers"].items():
        speaker = raw_name.strip()
        if not speaker:
            raise ValueError("Speaker config contains an empty speaker name.")
        if not isinstance(entry, dict) or set(entry) != {"voice"}:
            raise ValueError(f"Speaker {speaker!r} must contain exactly one 'voice' string.")
        voice = entry["voice"]
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError(f"Speaker {speaker!r} must contain a nonempty 'voice' string.")
        if speaker in result:
            raise ValueError(f"Speaker config contains duplicate normalized speaker {speaker!r}.")
        result[speaker] = voice.strip()
    return result


def _matching_voice_names(
    voices: list[dict[str, Any]], gender: str, locale: str | None, language: str | None,
) -> list[str]:
    """Filter current Edge voice metadata for automatic dialogue assignment."""
    requested_gender = gender.casefold()
    requested_locale = locale.casefold() if locale else None
    requested_language = language.casefold() if language else None
    candidates: list[str] = []
    for voice in voices:
        short_name = voice.get("ShortName")
        voice_gender = voice.get("Gender")
        voice_locale = voice.get("Locale")
        status = voice.get("Status", "")
        if not all(isinstance(value, str) for value in (short_name, voice_gender, voice_locale)):
            continue
        if isinstance(status, str) and status.casefold() == "deprecated":
            continue
        if voice_gender.casefold() != requested_gender:
            continue
        if requested_locale and voice_locale.casefold() != requested_locale:
            continue
        if requested_language and voice_locale.split("-", 1)[0].casefold() != requested_language:
            continue
        candidates.append(short_name)
    return sorted(candidates, key=str.casefold)


def resolve_auto_voice_map(
    speakers: list[str],
    voices: list[dict[str, Any]],
    gender: str,
    locale: str | None = None,
    language: str | None = None,
    seed: int | None = None,
) -> dict[str, str]:
    """Choose one distinct current Edge voice per speaker without replacement."""
    if gender.casefold() not in {"male", "female"}:
        raise ValueError("Speaker gender must be 'male' or 'female'.")
    if bool(locale) == bool(language):
        raise ValueError("Automatic speaker assignment needs exactly one of locale or language.")

    unique_speakers = list(dict.fromkeys(speakers))
    candidates = _matching_voice_names(voices, gender, locale, language)
    if len(candidates) < len(unique_speakers):
        filter_description = f"gender={gender}"
        if locale:
            filter_description += f", locale={locale}"
        else:
            filter_description += f", language={language}"
        raise ValueError(
            f"Not enough voices matching {filter_description}: {len(unique_speakers)} speakers, "
            f"{len(candidates)} available. Use --speaker-config or broaden the locale/language filter."
        )

    random.Random(seed).shuffle(candidates)
    return dict(zip(unique_speakers, candidates))


def resolve_config_voice_map(
    speakers: list[str], config: dict[str, str], voices: list[dict[str, Any]] | None,
) -> dict[str, str]:
    """Validate a config mapping and return the selected voice for each speaker."""
    unique_speakers = list(dict.fromkeys(speakers))
    missing = [speaker for speaker in unique_speakers if speaker not in config]
    if missing:
        raise ValueError(f"Speaker config is missing: {', '.join(missing)}")

    extra = [speaker for speaker in config if speaker not in unique_speakers]
    if extra:
        warnings.warn(
            f"Speaker config entries are unused: {', '.join(extra)}.",
            UserWarning,
            stacklevel=2,
        )

    if voices is not None:
        available = {
            voice["ShortName"]: voice
            for voice in voices
            if isinstance(voice.get("ShortName"), str)
        }
        unavailable = [config[speaker] for speaker in unique_speakers if config[speaker] not in available]
        if unavailable:
            raise ValueError(f"Configured voice(s) unavailable: {', '.join(dict.fromkeys(unavailable))}")
        deprecated = []
        for speaker in unique_speakers:
            status = available[config[speaker]].get("Status", "")
            if isinstance(status, str) and status.casefold() == "deprecated":
                deprecated.append(config[speaker])
        if deprecated:
            raise ValueError(
                f"Configured voice(s) deprecated: {', '.join(dict.fromkeys(deprecated))}"
            )
    return {speaker: config[speaker] for speaker in unique_speakers}


def fetch_voices() -> list[dict[str, Any]]:
    """Fetch Edge voice metadata once, supporting edge-tts API return variants."""
    result = edge_tts.list_voices()
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    if not isinstance(result, list):
        raise RuntimeError("Edge TTS returned an invalid voice list.")
    return result


__all__ = [
    "DialogueTurn",
    "fetch_voices",
    "load_speaker_config",
    "parse_dialogue",
    "resolve_auto_voice_map",
    "resolve_config_voice_map",
]
