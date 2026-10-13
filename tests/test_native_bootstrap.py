"""Lazy native runtime acquisition through the same verified XDG asset boundary."""

import hashlib
import io
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import md2audio_models


def _payload(entries: list[tuple[str, bytes | str]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as archive:
        for name, content in entries:
            member = tarfile.TarInfo("test-runtime/" + name)
            if isinstance(content, str):
                member.type = tarfile.SYMTYPE
                member.linkname = content
                archive.addfile(member)
            else:
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    return output.getvalue()


@pytest.fixture
def native(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    files = {"bin/sherpa-onnx-offline-tts": b"native fixture", "lib/libonnxruntime.so": b"library fixture"}
    payload = _payload(list(files.items()))
    metadata = ("test-runtime", len(payload), hashlib.sha256(payload).hexdigest(), 4096,
                {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in files.items()})
    monkeypatch.setattr(md2audio_models, "_NATIVE", {("linux", "x86_64"): metadata,
                                                   ("android", "aarch64"): metadata}, raising=False)
    monkeypatch.setattr(md2audio_models, "platform", platform, raising=False)
    monkeypatch.setattr(md2audio_models, "shutil", shutil, raising=False)
    monkeypatch.setattr(md2audio_models, "subprocess", subprocess, raising=False)
    monkeypatch.setattr(md2audio_models.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="30\n", returncode=0))
    monkeypatch.setattr(md2audio_models.sys, "platform", "linux")
    monkeypatch.delattr(md2audio_models.sys, "getandroidapilevel", raising=False)
    monkeypatch.setattr(md2audio_models.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(md2audio_models.shutil, "which", lambda name: None)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    requests: list[str] = []

    def response(url: str, timeout: float):
        requests.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(md2audio_models, "build_opener", lambda *args: SimpleNamespace(open=response))
    return files, payload, requests


def test_native_first_use_downloads_selected_binary_and_library_then_reuses_offline(
    native, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    files, _, requests = native
    before_path = os.environ["PATH"]
    executable = tmp_path / "data/md2audio/native/test-runtime/sherpa-onnx-offline-tts"
    assert md2audio_models.ensure_native() == str(executable)
    assert requests == ["https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.8/test-runtime.tar.bz2"]
    for name, data in files.items():
        path = executable.parent / Path(name).name
        assert path.read_bytes() == data
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path == executable else 0o600)
    assert not list(executable.parent.parent.glob(".test-runtime.download-*"))
    monkeypatch.setattr(md2audio_models, "build_opener", lambda *args: pytest.fail("Cached runtime contacted network"))
    assert md2audio_models.ensure_native() == str(executable)
    assert os.environ["PATH"] == before_path


def test_existing_path_native_runtime_needs_no_cache_platform_or_network(native, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(md2audio_models.shutil, "which", lambda name: "/already-installed/sherpa-onnx-offline-tts")
    monkeypatch.setattr(md2audio_models.platform, "machine", lambda: "unsupported")
    monkeypatch.setattr(md2audio_models, "build_opener", lambda *args: pytest.fail("Existing PATH runtime contacted network"))
    assert md2audio_models.ensure_native() == "/already-installed/sherpa-onnx-offline-tts"
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("platform_name,machine,android_api,valid", [
    ("linux", "x86_64", False, True), ("android", "aarch64", False, True),
    ("linux", "aarch64", True, True), ("linux", "aarch64", False, False),
    ("android", "armv7l", False, False),
])
def test_native_platform_selection_includes_termux_linux_python(
    native, tmp_path, monkeypatch, platform_name, machine, android_api, valid,
) -> None:
    monkeypatch.setattr(md2audio_models.sys, "platform", platform_name)
    monkeypatch.setattr(md2audio_models.platform, "machine", lambda: machine)
    if android_api:
        monkeypatch.setattr(md2audio_models.sys, "getandroidapilevel", lambda: 24, raising=False)
    if valid:
        assert Path(md2audio_models.ensure_native()).is_file()
    else:
        with pytest.raises(RuntimeError, match="Unsupported.*platform"):
            md2audio_models.ensure_native()
        assert not (tmp_path / "data").exists()
        assert native[2] == []


@pytest.mark.parametrize("sdk", ["26", "unknown", ""])
def test_android_runtime_sdk_must_support_bundled_onnx_before_any_download(native, tmp_path, monkeypatch, sdk) -> None:
    monkeypatch.setattr(md2audio_models.sys, "platform", "android")
    monkeypatch.setattr(md2audio_models.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(md2audio_models.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=sdk, returncode=0))
    with pytest.raises(RuntimeError, match="Android.*27"):
        md2audio_models.ensure_native()
    assert native[2] == [] and not (tmp_path / "data").exists()


@pytest.mark.parametrize("damage", ["content", "symlink", "permissions", "not-executable"])
def test_invalid_cached_native_runtime_is_preserved_and_never_downloaded(native, monkeypatch, damage) -> None:
    executable = Path(md2audio_models.ensure_native())
    library = executable.parent / "libonnxruntime.so"
    if damage == "content":
        library.write_bytes(b"corrupt")
    elif damage == "symlink":
        library.unlink()
        library.symlink_to(executable)
    elif damage == "permissions":
        executable.chmod(0o722)
    else:
        executable.chmod(0o600)
    metadata = (executable.lstat(), library.lstat())
    monkeypatch.setattr(md2audio_models, "build_opener", lambda *args: pytest.fail("Corrupt native cache contacted network"))
    with pytest.raises(RuntimeError, match="cached|executable"):
        md2audio_models.ensure_native()
    assert (executable.lstat(), library.lstat()) == metadata


@pytest.mark.parametrize("entry", [("../escape", b"bad"), ("link", "../../escape"),
                                   ("bin/sherpa-onnx-offline-tts", b"duplicate")])
def test_native_archive_traversal_links_and_duplicates_do_not_publish(native, tmp_path, monkeypatch, entry) -> None:
    files, _, _ = native
    payload = _payload([*files.items(), entry])
    old = md2audio_models._NATIVE[("linux", "x86_64")]
    monkeypatch.setitem(md2audio_models._NATIVE, ("linux", "x86_64"),
                        (old[0], len(payload), hashlib.sha256(payload).hexdigest(), *old[3:]))
    monkeypatch.setattr(md2audio_models, "build_opener", lambda *args: SimpleNamespace(
        open=lambda *args, **kwargs: io.BytesIO(payload)))
    with pytest.raises(RuntimeError, match="archive"):
        md2audio_models.ensure_native()
    root = tmp_path / "data/md2audio/native"
    assert not (root / "test-runtime").exists()
    assert not list(root.glob(".test-runtime.download-*"))
    assert not (tmp_path / "escape").exists()


def test_failed_native_download_never_publishes_partial_runtime(native, tmp_path, monkeypatch) -> None:
    payload = native[1][:-10]
    monkeypatch.setattr(md2audio_models, "build_opener", lambda *args: SimpleNamespace(
        open=lambda *args, **kwargs: io.BytesIO(payload)))
    with pytest.raises(RuntimeError, match="SHA256|size"):
        md2audio_models.ensure_native()
    root = tmp_path / "data/md2audio/native"
    assert not (root / "test-runtime").exists()
    assert not list(root.glob(".test-runtime.download-*"))


def test_native_concurrent_cold_uses_share_one_download_and_atomic_ready_directory(native, monkeypatch) -> None:
    entered = threading.Event()
    contended = threading.Event()
    release = threading.Event()
    results: list[str] = []
    errors: list[BaseException] = []
    payload, requests = native[1:]
    original_flock = md2audio_models.fcntl.flock

    def flock(*args):
        try:
            return original_flock(*args)
        except BlockingIOError:
            contended.set()
            raise

    def response(url, timeout):
        requests.append(url)
        entered.set()
        assert release.wait(3), "Concurrent test did not release download"
        return io.BytesIO(payload)

    def acquire():
        try:
            results.append(md2audio_models.ensure_native())
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(md2audio_models.fcntl, "flock", flock)
    monkeypatch.setattr(md2audio_models, "build_opener", lambda *args: SimpleNamespace(open=response))
    first = threading.Thread(target=acquire)
    second = threading.Thread(target=acquire)
    first.start()
    try:
        assert entered.wait(2)
        second.start()
        assert contended.wait(2), "Second acquisition did not contend for the runtime lock"
    finally:
        release.set()
        first.join(3)
        if second.ident is not None:
            second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert errors == [] and len(results) == 2 and results[0] == results[1]
    assert len(requests) == 1


def test_native_safe_xdg_directory_symlink_is_retained(native, tmp_path, monkeypatch) -> None:
    real = tmp_path / "real-data"
    real.mkdir()
    link = tmp_path / "linked-data"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(link))
    executable = Path(md2audio_models.ensure_native())
    assert executable.is_relative_to(real) and link.is_symlink()


@pytest.mark.parametrize("separator", [":", ";", "$ORIGIN"])
def test_native_cache_directory_must_be_representable_in_child_library_search_path(
    native, tmp_path, monkeypatch, separator,
) -> None:
    real = tmp_path / ("private" + separator + "data")
    real.mkdir()
    link = tmp_path / "configured-data"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(link))
    with pytest.raises(RuntimeError, match="library search path"):
        md2audio_models.ensure_native()
    assert native[2] == [] and link.is_symlink()
    assert not list(real.glob("md2audio/native/.test-runtime*"))


@pytest.mark.parametrize("suffix", ["safe", ":", ";", "$ORIGIN"])
def test_cached_native_version_symlink_validates_its_final_library_directory(
    native, tmp_path, monkeypatch, suffix,
) -> None:
    original = Path(md2audio_models.ensure_native()).parent
    target = tmp_path / ("private" + suffix + "runtime")
    original.rename(target)
    original.symlink_to(target, target_is_directory=True)

    def snapshot():
        result = {}
        for path in [original, target, *target.iterdir()]:
            metadata = path.lstat()
            result[str(path)] = (metadata.st_ino, metadata.st_mode, metadata.st_size,
                                 metadata.st_mtime_ns, metadata.st_ctime_ns,
                                 path.read_bytes() if path.is_file() else None)
        return result

    before = snapshot()
    monkeypatch.setattr(md2audio_models, "build_opener", lambda *args: pytest.fail("Cached symlink contacted network"))
    if suffix == "safe":
        assert md2audio_models.ensure_native() == str(target / "sherpa-onnx-offline-tts")
    else:
        with pytest.raises(RuntimeError, match="library search path"):
            md2audio_models.ensure_native()
    assert snapshot() == before and original.is_symlink()
    assert len(native[2]) == 1


@pytest.mark.parametrize("case", [
    "sticky-origin", "sticky-sibling-parent", "sticky-third-parent",
    "sibling-alias", "third-alias", "extra-origin", "writable-sibling-file", "linked-third-file",
    "private-control", "sticky-readonly-control", "safe-version-alias", "safe-sibling-alias",
])
def test_native_loader_namespaces_reject_peer_candidates_without_mutating_cache(
    native, tmp_path, monkeypatch, case,
) -> None:
    executable = Path(md2audio_models.ensure_native())
    origin = executable.parent
    sibling = origin.parent / "lib"
    third = origin.parents[2] / "sherpa_onnx"
    if case == "sticky-origin":
        origin.chmod(0o1777)
    elif case in {"sticky-sibling-parent", "safe-version-alias"}:
        shared = tmp_path / "shared"
        shared.mkdir(mode=0o700)
        target = shared / "release"
        origin.rename(target)
        origin.symlink_to(os.path.relpath(target, origin.parent), target_is_directory=True)
        if case == "sticky-sibling-parent":
            shared.chmod(0o1777)
    elif case in {"sticky-third-parent", "sticky-readonly-control"}:
        origin.parents[2].chmod(0o1777 if case == "sticky-third-parent" else 0o1755)
    elif case in {"sibling-alias", "third-alias", "safe-sibling-alias"}:
        target = tmp_path / "library-target"
        target.mkdir(mode=0o700 if case == "safe-sibling-alias" else 0o777)
        target.chmod(0o700 if case == "safe-sibling-alias" else 0o777)
        (third if case == "third-alias" else sibling).symlink_to(target, target_is_directory=True)
    elif case == "extra-origin":
        (origin / "libstdc++.so.6").write_bytes(b"unpinned dependency")
    elif case == "writable-sibling-file":
        sibling.mkdir(mode=0o700)
        library = sibling / "libstdc++.so.6"
        library.write_bytes(b"peer writable dependency")
        library.chmod(0o666)
    elif case == "linked-third-file":
        (third / "lib").mkdir(parents=True, mode=0o700)
        (third / "lib/libstdc++.so.6").symlink_to(executable)

    def snapshot():
        result = {}
        for path in tmp_path.rglob("*"):
            metadata = path.lstat()
            result[str(path)] = (metadata.st_ino, metadata.st_uid, metadata.st_gid, metadata.st_mode,
                                 metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns,
                                 path.read_bytes() if path.is_file() else None)
        return result

    before = snapshot()
    monkeypatch.setattr(md2audio_models, "build_opener", lambda *args: pytest.fail("Loader validation contacted network"))
    if case.endswith("control") or case.startswith("safe-"):
        assert md2audio_models.ensure_native() == str(origin.resolve() / executable.name)
    else:
        with pytest.raises(RuntimeError, match="library|runtime"):
            md2audio_models.ensure_native()
    assert snapshot() == before and len(native[2]) == 1


@pytest.mark.parametrize("case", ["sticky-xdg", "unsafe-third-alias"])
def test_unsafe_native_loader_namespace_is_refused_before_first_download(native, tmp_path, case) -> None:
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    if case == "sticky-xdg":
        data.chmod(0o1777)
    else:
        target = tmp_path / "unsafe"
        target.mkdir(mode=0o777)
        target.chmod(0o777)
        (data / "sherpa_onnx").symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="library"):
        md2audio_models.ensure_native()
    assert native[2] == []
    assert not (data / "md2audio/native/test-runtime").exists()
    assert not list(data.glob("md2audio/native/.test-runtime.download-*"))
    assert not (data / "sherpa_onnx/lib").exists()


def test_native_failed_acquisition_keeps_original_cli_mp3(tmp_path, monkeypatch) -> None:
    import md2audio
    source = tmp_path / "input.md"
    source.write_text("你好。", encoding="utf-8")
    output = tmp_path / "audio.mp3"
    output.write_bytes(b"last good audio")

    def unavailable():
        raise md2audio_models.LocalTTSError("Native download unavailable")

    monkeypatch.setattr(md2audio_models, "ensure_native", unavailable)
    monkeypatch.setattr(md2audio_models, "ensure_model", lambda name: pytest.fail("Failed runtime downloaded model"))
    monkeypatch.setattr(md2audio.shutil, "which", lambda name: "/available/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr(sys, "argv", ["md2audio", str(source), "--local-tts", "-o", str(output)])
    with pytest.raises(SystemExit) as exc:
        md2audio.main()
    assert exc.value.code == 1 and output.read_bytes() == b"last good audio"


@pytest.mark.parametrize("mask", [0o000, 0o002])
def test_new_native_files_remain_private_without_changing_process_umask(native, mask) -> None:
    previous = os.umask(mask)
    try:
        executable = Path(md2audio_models.ensure_native())
        assert stat.S_IMODE(executable.stat().st_mode) == 0o700
        assert stat.S_IMODE((executable.parent / "libonnxruntime.so").stat().st_mode) == 0o600
        observed = os.umask(mask)
        assert observed == mask
    finally:
        os.umask(previous)
