"""Download two pinned VITS models lazily into persistent XDG user data."""

import fcntl
import hashlib
import os
import stat
import sys
import tarfile
import tempfile
import time
from collections import deque
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from urllib.request import HTTPRedirectHandler, build_opener

from md2audio_local import LocalTTSError

__all__ = ["ensure_model"]

_LOCK_TIMEOUT_SECONDS = 600
_DOWNLOAD_TIMEOUT_SECONDS = 600
_READ_BYTES = 1024 * 1024
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


class _HTTPSRedirect(HTTPRedirectHandler):
    """Keep model downloads encrypted through GitHub's asset redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise LocalTTSError("Model download refused an insecure redirect")
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
        android = sys.platform == "android"
        system_path = android and candidate in {Path("/data"), Path("/data/data"),
                                                Path("/data/user"), Path("/data/user/0")}
        trusted_owner = metadata.st_uid in {0, os.getuid()} or (system_path and metadata.st_uid == metadata.st_gid == 1000)
        if not trusted_owner:
            raise LocalTTSError(f"Unsafe model directory owner: {candidate}")
        if stat.S_ISLNK(metadata.st_mode):
            links += 1
            if links > 40:
                raise LocalTTSError(f"Too many model directory symlinks: {path}")
            target = Path(os.readlink(candidate))
            if not candidate.exists():
                raise LocalTTSError(f"Unsafe model directory dangling symlink: {candidate}")
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
            raise LocalTTSError(f"Unsafe model directory permissions or type: {candidate}")
        current = candidate
    return current


def _verify_files(directory: Path, files: dict[str, tuple[int, str]]) -> None:
    """Verify bounded contents through no-follow file descriptors before use or promotion."""
    for name, (size, expected) in files.items():
        with os.fdopen(os.open(directory / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as source:
            metadata = os.fstat(source.fileno())
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid not in {0, os.getuid()} or
                    metadata.st_size != size or metadata.st_mode & 0o022):
                raise LocalTTSError(f"Invalid cached model file: {directory / name}")
            digest = hashlib.sha256()
            count = 0
            while data := source.read(min(_READ_BYTES, size - count + 1)):
                count += len(data)
                if count > size:
                    raise LocalTTSError(f"Oversized cached model file: {directory / name}")
                digest.update(data)
            if count != size or digest.hexdigest() != expected:
                raise LocalTTSError(f"Corrupt cached model file: {directory / name}")


def _download(url: str, output: Path, size: int, expected: str) -> None:
    """Stream a bounded HTTPS asset with both socket and overall deadlines."""
    print(f"Downloading local TTS model ({size / 1_000_000:.1f} MB)...", file=sys.stderr, flush=True)
    deadline = time.monotonic() + _DOWNLOAD_TIMEOUT_SECONDS
    next_progress = time.monotonic() + 5
    digest = hashlib.sha256()
    count = 0
    with build_opener(_HTTPSRedirect()).open(url, timeout=30) as response, output.open("xb") as target:
        while True:
            if time.monotonic() >= deadline:
                raise LocalTTSError("Model download timed out; retry --local-tts when connectivity improves")
            data = response.read1(min(_READ_BYTES, size - count + 1))
            if not data:
                break
            count += len(data)
            if count > size:
                raise LocalTTSError("Model archive exceeds its pinned size")
            target.write(data)
            digest.update(data)
            if time.monotonic() >= next_progress:
                print(f"  {count / 1_000_000:.1f}/{size / 1_000_000:.1f} MB", file=sys.stderr, flush=True)
                next_progress = time.monotonic() + 5
    if count != size or digest.hexdigest() != expected:
        raise LocalTTSError("Model archive size or SHA256 mismatch; retry with a trusted network connection")


def _extract(archive: Path, destination: Path, top: str, expanded_limit: int,
             files: dict[str, tuple[int, str]]) -> None:
    """Reject unsafe members and copy only pinned regular model files into owned staging."""
    seen: set[str] = set()
    expanded = 0
    deadline = time.monotonic() + _DOWNLOAD_TIMEOUT_SECONDS
    with tarfile.open(archive, "r|bz2") as source:
        for member in source:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != top or
                    member.name in seen or len(seen) >= 64 or not (member.isfile() or member.isdir())):
                raise LocalTTSError(f"Unsafe model archive member: {member.name}")
            seen.add(member.name)
            expanded += member.size
            if member.size < 0 or expanded > expanded_limit or time.monotonic() >= deadline:
                raise LocalTTSError("Model archive exceeds extraction limits")
            if len(name.parts) != 2 or name.name not in files or not member.isfile():
                continue
            if member.size != files[name.name][0]:
                raise LocalTTSError(f"Unexpected model archive file size: {member.name}")
            stream = source.extractfile(member)
            if stream is None:
                raise LocalTTSError(f"Missing model archive data: {member.name}")
            with stream, (destination / name.name).open("xb") as target:
                os.fchmod(target.fileno(), 0o600)
                remaining = member.size
                while remaining:
                    if time.monotonic() >= deadline:
                        raise LocalTTSError("Model archive extraction timed out")
                    data = stream.read(min(_READ_BYTES, remaining))
                    if not data:
                        raise LocalTTSError(f"Truncated model archive file: {member.name}")
                    remaining -= len(data)
                    target.write(data)
    _verify_files(destination, files)


def ensure_model(model: str = "melo") -> Path:
    """Resolve a verified built-in model, downloading it only on first local synthesis.

    Uses absolute XDG_DATA_HOME or ~/.local/share, under md2audio/models. Concurrent
    first uses share a bounded OS lock and atomic publication. Each reuse verifies
    pinned file sizes/hashes without network. Existing corrupt data is preserved and
    reported; only this invocation's unpublished staging is removed on failure.
    sherpa-onnx-offline-tts and FFmpeg installation is the caller's responsibility.
    """
    if model not in _MODELS:
        raise LocalTTSError(f"Unknown local model: {model}; choose melo or aishell3")
    top, archive_size, archive_hash, expanded_limit, files = _MODELS[model]
    value = os.environ.get("XDG_DATA_HOME", "")
    data_home = Path(value) if value and Path(value).is_absolute() else Path.home() / ".local/share"
    try:
        root = _safe_directory(data_home / "md2audio/models")
        lock_fd = os.open(root / f".{top}.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock_fd, "rb") as lock:
            metadata = os.fstat(lock.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o022:
                raise LocalTTSError(f"Unsafe model lock file: {root / f'.{top}.lock'}")
            deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise LocalTTSError("Timed out waiting for local model download lock; retry later")
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            directory = root / top
            if directory.exists() or directory.is_symlink():
                directory = _safe_directory(directory)
                try:
                    _verify_files(directory, files)
                except (OSError, LocalTTSError) as exc:
                    raise LocalTTSError(f"Incomplete or corrupt cached model at {directory}; "
                                        f"rename that directory to preserve it, then retry: {exc}") from exc
                return directory
            with tempfile.TemporaryDirectory(prefix=f".{top}.download-", dir=root) as work:
                archive = Path(work) / "model.tar.bz2"
                staged = Path(work) / top
                staged.mkdir(mode=0o700)
                url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{top}.tar.bz2"
                _download(url, archive, archive_size, archive_hash)
                print("Verifying and unpacking local TTS model...", file=sys.stderr, flush=True)
                _extract(archive, staged, top, expanded_limit, files)
                staged.rename(directory)
            print(f"Local TTS model ready: {directory}", file=sys.stderr, flush=True)
            return directory
    except (OSError, HTTPException, tarfile.TarError, EOFError) as exc:
        raise LocalTTSError(f"Could not prepare local model in {data_home}: {exc}; retry --local-tts") from exc
