"""Acquire pinned VITS models and native TTS runtimes lazily in XDG user data."""

import fcntl
import hashlib
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import deque
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from urllib.request import HTTPRedirectHandler, build_opener

__all__ = ["LocalTTSError", "ensure_model", "ensure_native"]

_LOCK_TIMEOUT_SECONDS = 600
_DOWNLOAD_TIMEOUT_SECONDS = 600
_READ_BYTES = 1024 * 1024


class LocalTTSError(RuntimeError):
    """A local dependency, model, or synthesis process could not complete."""


# Pins were measured from complete official HTTPS assets, not publisher-signed digests.
_MODELS: dict[str, tuple[str, int, str, int, dict[str, tuple[int, str]]]] = {
    "melo": ("vits-melo-tts-zh_en", 167006755,
        "e58351ed7149f290a54534538badd4077cdbe6fddc964b24d0bee870415d1514", 191246256, {
            "model.onnx": (170429550, "bf30582eb1b012250a35b1a4a80e7dfbcf8485e7bb9de0d95efbbeef0e4ad86d"),
            "LICENSE": (1053, "88a50e5a02bbc2a5c2f084dc19da751aa97b1690f5fda76cd8005c8634d1ca70"),
            "tokens.txt": (655, "d18664a7e12bd7ea1022ddaf951e534e136815016c5a809d6b64156bffb4369d"),
            "README.md": (257, "ae6e44dead53dfdbd5b4a77106226f3118eb910222b46cb5533a3860773fc6c7"),
            "date.fst": (59154, "eb8aa079ae3cb81d8f4404992f39d61a0cb990947512b5b8d1e54d1f6980e718"),
            "phone.fst": (88630, "1ac2b6fa56b1442320c4de7db08353bab8963a2b57f365eebcdd3a2d3562f8d7"),
            "lexicon.txt": (6837671, "7236884b02435ac5d10cf69b4be40a61b45aa676b5300f0e412f185748fee528"),
            "number.fst": (64482, "743f402181fcfebf76cc2f0546b71fa26476e626fbe4e460fb7b4c3a7a8bd5bd"),
        }),
    "aishell3": ("vits-icefall-zh-aishell3", 31559701,
        "ab468db3a3308cdd861495e0db2f25d79418a0c00639f74944c7cdf5dd8c6ec1", 213479522, {
            "date.fst": (59154, "eb8aa079ae3cb81d8f4404992f39d61a0cb990947512b5b8d1e54d1f6980e718"),
            "lexicon.txt": (2042943, "ab2e61d357551e7b24ddd965d924aca784c20165ff58c150794e539c6b5e9e35"),
            "number.fst": (64482, "743f402181fcfebf76cc2f0546b71fa26476e626fbe4e460fb7b4c3a7a8bd5bd"),
            "speakers.txt": (1392, "50a825fc9a89172eee04a4d68869d933d0eca0d847f3b6fcffff9514cc72442d"),
            "model.onnx": (30482262, "5511d651b7840c0a93a6bbfd4afd070a2c7f39ca1ec3ff2ecd73191519bbb852"),
            "phone.fst": (88630, "1ac2b6fa56b1442320c4de7db08353bab8963a2b57f365eebcdd3a2d3562f8d7"),
            "tokens.txt": (1671, "50b45a7b7de1752fd3c7b4755661c285f1547f59186eca2281089a81307ad953"),
        }),
}

_NATIVE = {
    ("linux", "x86_64"): ("sherpa-onnx-v1.13.8-linux-x64-shared", 28156791,
        "c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4", 86050728, {
            "bin/sherpa-onnx-offline-tts": (2551488, "dde4fbd93181de356d9dc32cd734407be0da64c10ad5883a5a1a1328e9fbc879"),
            "lib/libonnxruntime.so": (27026609, "4b3607aebd1784b26b6f9b20e4bd974c7ab8287043e4d095cb7d2cb40b5e566e"),
        }),
    ("android", "aarch64"): ("sherpa-onnx-v1.13.8-android-aarch64-termux-shared", 16451109,
        "ad7bac42adc6295293fe6d36475936afb227cdc757cf293c02a706f6aca4e0a2", 46043231, {
            "bin/sherpa-onnx-offline-tts": (1957768, "4620ba519b24d01ef1d08b1be2d33cf9e21a93593ae23adb7fc23527b9ad5e81"),
            "lib/libonnxruntime.so": (22249560, "33847ad43bffe204699fd4a27f7f3603452a8cdaf2f9a44983a0bc31ffcf2da1"),
        }),
}


class _HTTPSRedirect(HTTPRedirectHandler):
    """Keep asset downloads encrypted through GitHub's redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise LocalTTSError("Local TTS download refused an insecure redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _safe_directory(path: Path) -> Path:
    """Admit every namespace hop and create private missing directories; retain symlinks."""
    current = Path(path.anchor)
    pending = deque(path.parts[1:])
    links = 0
    android_private: Path | None = None
    while pending:
        component = pending.popleft()
        if component == "..":
            current = current.parent
            if android_private is not None and not current.is_relative_to(android_private):
                android_private = None
            continue
        candidate = current / component
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            try:
                candidate.mkdir(mode=0o700)
            except FileExistsError:
                pass  # Another cold start may have created this same component.
            metadata = candidate.lstat()
        android = sys.platform == "android" or hasattr(sys, "getandroidapilevel")
        system_path = android and candidate in {Path("/data"), Path("/data/data"),
                                                Path("/data/user"), Path("/data/user/0")}
        trusted_owner = metadata.st_uid in {0, os.getuid()} or (system_path and metadata.st_uid == metadata.st_gid == 1000)
        if not trusted_owner:
            raise LocalTTSError(f"Unsafe local TTS directory owner: {candidate}")
        if stat.S_ISLNK(metadata.st_mode):
            links += 1
            if links > 40:
                raise LocalTTSError(f"Too many local TTS directory symlinks: {path}")
            target = Path(os.readlink(candidate))
            if not candidate.exists():
                raise LocalTTSError(f"Unsafe local TTS directory dangling symlink: {candidate}")
            if target.is_absolute():
                current = Path(target.anchor)
                android_private = None
                pending.extendleft(reversed(target.parts[1:]))
            else:
                pending.extendleft(reversed(target.parts))
            continue
        if (android and android_private is None and metadata.st_uid == os.getuid() and
                os.getuid() >= 10000 and not metadata.st_mode & 0o077):
            android_private = candidate
        android_group = android and metadata.st_uid == metadata.st_gid and (
            (metadata.st_uid == os.getuid() == os.getgid() and android_private is not None and
             candidate.is_relative_to(android_private)) or
            (system_path and metadata.st_uid == metadata.st_gid == 1000))
        unsafe_mode = metadata.st_mode & 0o022 and not metadata.st_mode & stat.S_ISVTX
        if (not stat.S_ISDIR(metadata.st_mode) or
                (unsafe_mode and not (android_group and not metadata.st_mode & 0o002))):
            raise LocalTTSError(f"Unsafe local TTS directory permissions or type: {candidate}")
        current = candidate
    return current


def _verify_files(directory: Path, files: dict[str, tuple[int, str]], executable: str | None = None) -> None:
    """Verify bounded contents through no-follow file descriptors before use or promotion."""
    for name, (size, expected) in files.items():
        name = PurePosixPath(name).name
        with os.fdopen(os.open(directory / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as source:
            metadata = os.fstat(source.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid not in {0, os.getuid()} or
                    metadata.st_size != size or metadata.st_mode & 0o022):
                raise LocalTTSError(f"Invalid cached local TTS file: {directory / name}")
            digest = hashlib.sha256()
            count = 0
            while data := source.read(min(_READ_BYTES, size - count + 1)):
                count += len(data)
                if count > size:
                    raise LocalTTSError(f"Oversized cached local TTS file: {directory / name}")
                digest.update(data)
            if count != size or digest.hexdigest() != expected:
                raise LocalTTSError(f"Corrupt cached local TTS file: {directory / name}")
            if name == executable and not metadata.st_mode & stat.S_IXUSR:
                raise LocalTTSError(f"Cached local TTS runtime is not executable: {directory / name}")


def _download(url: str, output: Path, size: int, expected: str, label: str) -> None:
    """Stream a bounded HTTPS asset with both socket and overall deadlines."""
    print(f"Downloading local TTS {label} ({size / 1_000_000:.1f} MB)...", file=sys.stderr, flush=True)
    deadline = time.monotonic() + _DOWNLOAD_TIMEOUT_SECONDS
    next_progress = time.monotonic() + 5
    digest = hashlib.sha256()
    count = 0
    with build_opener(_HTTPSRedirect()).open(url, timeout=30) as response, output.open("xb") as target:
        while True:
            if time.monotonic() >= deadline:
                raise LocalTTSError("Local TTS download timed out; retry --local-tts when connectivity improves")
            data = response.read1(min(_READ_BYTES, size - count + 1))
            if not data:
                break
            count += len(data)
            if count > size:
                raise LocalTTSError("Local TTS archive exceeds its pinned size")
            target.write(data)
            digest.update(data)
            if time.monotonic() >= next_progress:
                print(f"  {count / 1_000_000:.1f}/{size / 1_000_000:.1f} MB", file=sys.stderr, flush=True)
                next_progress = time.monotonic() + 5
    if count != size or digest.hexdigest() != expected:
        raise LocalTTSError("Local TTS archive size or SHA256 mismatch; retry with a trusted network connection")


def _extract(archive: Path, destination: Path, top: str, expanded_limit: int,
             files: dict[str, tuple[int, str]], executable: str | None) -> None:
    """Reject unsafe members and flatten only pinned regular files into owned staging."""
    seen: set[str] = set()
    expanded = 0
    deadline = time.monotonic() + _DOWNLOAD_TIMEOUT_SECONDS
    with tarfile.open(archive, "r|bz2") as source:
        for member in source:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != top or
                    member.name in seen or len(seen) >= 64 or not (member.isfile() or member.isdir())):
                raise LocalTTSError(f"Unsafe local TTS archive member: {member.name}")
            seen.add(member.name)
            expanded += member.size
            if member.size < 0 or expanded > expanded_limit or time.monotonic() >= deadline:
                raise LocalTTSError("Local TTS archive exceeds extraction limits")
            relative = str(PurePosixPath(*name.parts[1:]))
            if relative not in files or not member.isfile():
                continue
            if member.size != files[relative][0]:
                raise LocalTTSError(f"Unexpected local TTS archive file size: {member.name}")
            stream = source.extractfile(member)
            if stream is None:
                raise LocalTTSError(f"Missing local TTS archive data: {member.name}")
            with stream, (destination / name.name).open("xb") as target:
                os.fchmod(target.fileno(), 0o700 if name.name == executable else 0o600)
                remaining = member.size
                while remaining:
                    if time.monotonic() >= deadline:
                        raise LocalTTSError("Local TTS archive extraction timed out")
                    data = stream.read(min(_READ_BYTES, remaining))
                    if not data:
                        raise LocalTTSError(f"Truncated local TTS archive file: {member.name}")
                    remaining -= len(data)
                    target.write(data)
    _verify_files(destination, files, executable)


def ensure_model(model: str = "melo") -> Path:
    """Resolve a verified built-in model, downloading it only on first local synthesis.

    Uses absolute XDG_DATA_HOME or ~/.local/share, under md2audio/models. Concurrent
    first uses share a bounded OS lock and atomic publication. Each reuse verifies
    pinned file sizes/hashes without network. Existing corrupt data is preserved and
    reported; only this invocation's unpublished staging is removed on failure.
    Native runtime acquisition is independent; FFmpeg installation is the caller's responsibility.
    """
    if model not in _MODELS:
        raise LocalTTSError(f"Unknown local model: {model}; choose melo or aishell3")
    return _ensure_asset(*_MODELS[model], category="models", release="tts-models")


def ensure_native() -> str:
    """Reuse a PATH runtime or lazily acquire the official v1.13.8 native CLI.

    Automatic acquisition supports Linux x86_64 and ARM64 Termux (Android API 27+).
    Stores the pinned executable and ONNX library together under md2audio/native
    in absolute XDG_DATA_HOME or ~/.local/share. Each reuse verifies private files
    offline. Returns the absolute executable path, without changing PATH or any
    system installation. The caller must prioritize its adjacent library for the
    managed runtime's child process only; FFmpeg and Termux libc++ remain prerequisites.
    """
    installed = shutil.which("sherpa-onnx-offline-tts")
    if installed is not None:
        return installed
    system = "android" if sys.platform == "android" or hasattr(sys, "getandroidapilevel") else sys.platform
    key = (system, platform.machine())
    if key not in _NATIVE:
        raise LocalTTSError(f"Unsupported automatic local TTS runtime platform: {system}/{key[1]}; "
                            "install sherpa-onnx-offline-tts on PATH for this platform")
    if system == "android":
        try:
            sdk = subprocess.run(["/system/bin/getprop", "ro.build.version.sdk"], check=True,
                                 capture_output=True, text=True, timeout=2).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise LocalTTSError("Cannot determine Android runtime SDK; automatic local TTS requires API 27+") from exc
        if not sdk.isascii() or not sdk.isdecimal() or len(sdk) > 3 or int(sdk) < 27:
            raise LocalTTSError(f"Automatic local TTS requires Android API 27+; runtime SDK is {sdk or 'unknown'}")
    directory = _ensure_asset(*_NATIVE[key], category="native", release="v1.13.8",
                              executable="sherpa-onnx-offline-tts")
    return str(directory / "sherpa-onnx-offline-tts")


def _ensure_asset(top: str, archive_size: int, archive_hash: str, expanded_limit: int,
                  files: dict[str, tuple[int, str]], *, category: str, release: str,
                  executable: str | None = None) -> Path:
    """Share verified download, bounded locking, and atomic publication for both asset types."""
    label = "runtime" if executable else "model"
    value = os.environ.get("XDG_DATA_HOME", "")
    data_home = Path(value) if value and Path(value).is_absolute() else Path.home() / ".local/share"
    try:
        root = _safe_directory(data_home / "md2audio" / category)
        if executable and any(character in str(root) for character in (":", ";", "$")):
            raise LocalTTSError(f"Native cache directory cannot be represented in the child library search path: {root}; "
                                "':' and ';' separate directories, and '$' expands loader tokens")
        lock_fd = os.open(root / f".{top}.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock_fd, "rb") as lock:
            metadata = os.fstat(lock.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o022:
                raise LocalTTSError(f"Unsafe local TTS lock file: {root / f'.{top}.lock'}")
            deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise LocalTTSError("Timed out waiting for local TTS download lock; retry later")
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            directory = root / top
            if directory.exists() or directory.is_symlink():
                directory = _safe_directory(directory)
                try:
                    _verify_files(directory, files, executable)
                except (OSError, LocalTTSError) as exc:
                    raise LocalTTSError(f"Incomplete or corrupt cached {label} at {directory}; "
                                        f"rename that directory to preserve it, then retry: {exc}") from exc
                return directory
            with tempfile.TemporaryDirectory(prefix=f".{top}.download-", dir=root) as work:
                archive = Path(work) / "asset.tar.bz2"
                staged = Path(work) / top
                staged.mkdir(mode=0o700)
                url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/{release}/{top}.tar.bz2"
                _download(url, archive, archive_size, archive_hash, label)
                print(f"Verifying and unpacking local TTS {label}...", file=sys.stderr, flush=True)
                _extract(archive, staged, top, expanded_limit, files, executable)
                staged.rename(directory)
            print(f"Local TTS {label} ready: {directory}", file=sys.stderr, flush=True)
            return directory
    except (OSError, HTTPException, tarfile.TarError, EOFError) as exc:
        raise LocalTTSError(f"Could not prepare local {label} in {data_home}: {exc}; retry --local-tts") from exc
