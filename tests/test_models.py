"""First-use model installation contracts using tiny archives and an offline HTTP boundary."""

import hashlib
import io
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import md2audio


def _archive(entries: list[tuple[str, bytes | str]], top: str = "test-melo") -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:bz2") as archive:
        for name, content in entries:
            item = tarfile.TarInfo(name if name.startswith("/") else f"{top}/{name}")
            if isinstance(content, str):
                item.type = tarfile.SYMTYPE
                item.linkname = content
                archive.addfile(item)
            else:
                item.size = len(content)
                archive.addfile(item, io.BytesIO(content))
    return output.getvalue()


@pytest.fixture
def models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import md2audio_models
    data = {"model.onnx": b"real fixture model", "tokens.txt": b"tokens", "lexicon.txt": b"lexicon"}
    payload = _archive(list(data.items()))
    metadata = ("test-melo", len(payload), hashlib.sha256(payload).hexdigest(), 4096,
                {name: (len(content), hashlib.sha256(content).hexdigest()) for name, content in data.items()})
    monkeypatch.setattr(md2audio_models, "_MODELS", {"melo": metadata, "aishell3": metadata})
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    downloads: list[str] = []

    def response(url: str, timeout: float):
        downloads.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(md2audio_models, "build_opener", lambda *args: SimpleNamespace(open=response))
    return md2audio_models, data, payload, downloads


def test_first_use_downloads_to_xdg_and_reuses_complete_model_offline(models, tmp_path: Path, capsys) -> None:
    module, data, _, downloads = models
    expected = tmp_path / "data/md2audio/models/test-melo"
    assert not expected.exists()
    assert module.ensure_model("melo") == expected
    assert downloads == ["https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/test-melo.tar.bz2"]
    assert {name: (expected / name).read_bytes() for name in data} == data
    assert not list(expected.parent.glob(".test-melo.download-*"))
    assert "Downloading" in capsys.readouterr().err
    assert module.ensure_model("melo") == expected
    assert len(downloads) == 1


@pytest.mark.parametrize("xdg", [None, "", "relative/data"])
def test_xdg_unset_empty_and_relative_use_home_data(models, tmp_path: Path, monkeypatch, xdg) -> None:
    module, _, _, _ = models
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    if xdg is None:
        monkeypatch.delenv("XDG_DATA_HOME")
    else:
        monkeypatch.setenv("XDG_DATA_HOME", xdg)
    assert module.ensure_model("melo") == home / ".local/share/md2audio/models/test-melo"


def test_safe_data_directory_symlink_keeps_storage_layout(models, tmp_path: Path, monkeypatch) -> None:
    module, _, _, _ = models
    real = tmp_path / "real-data"
    real.mkdir()
    link = tmp_path / "data-link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(link))
    assert module.ensure_model("melo") == real / "md2audio/models/test-melo"
    assert link.is_symlink()


@pytest.mark.parametrize("entry", [("../escape", b"bad"), ("/absolute", b"bad"),
                                    ("linked", "../../escape"), ("tokens.txt", b"duplicate")])
def test_unsafe_archive_does_not_publish_or_write_outside_staging(models, tmp_path: Path, monkeypatch, entry) -> None:
    module, data, _, _ = models
    payload = _archive([*data.items(), entry])
    old = module._MODELS["melo"]
    monkeypatch.setitem(module._MODELS, "melo", (old[0], len(payload), hashlib.sha256(payload).hexdigest(), *old[3:]))
    monkeypatch.setattr(module, "build_opener", lambda *args: SimpleNamespace(open=lambda *args, **kwargs: io.BytesIO(payload)))
    with pytest.raises(RuntimeError, match="archive"):
        module.ensure_model("melo")
    assert not (tmp_path / "data/md2audio/models/test-melo").exists()
    assert not (tmp_path / "escape").exists()
    assert not list((tmp_path / "data/md2audio/models").glob(".test-melo.download-*"))


@pytest.mark.parametrize("corruption", ["digest", "truncated", "oversized", "missing"])
def test_download_integrity_and_completeness_fail_without_ready_directory(models, tmp_path, monkeypatch, corruption) -> None:
    module, data, payload, _ = models
    if corruption == "digest":
        payload = payload[:-1] + bytes([payload[-1] ^ 1])
    elif corruption == "truncated":
        payload = payload[:10]
    elif corruption == "oversized":
        payload += b"too much"
    else:
        payload = _archive([(name, content) for name, content in data.items() if name != "tokens.txt"])
        old = module._MODELS["melo"]
        monkeypatch.setitem(module._MODELS, "melo", (old[0], len(payload), hashlib.sha256(payload).hexdigest(), *old[3:]))
    monkeypatch.setattr(module, "build_opener", lambda *args: SimpleNamespace(open=lambda *args, **kwargs: io.BytesIO(payload)))
    with pytest.raises(RuntimeError):
        module.ensure_model("melo")
    assert not (tmp_path / "data/md2audio/models/test-melo").exists()
    assert not list((tmp_path / "data/md2audio/models").glob(".test-melo.download-*"))


def test_cancelled_download_removes_only_owned_temporary_state(models, tmp_path, monkeypatch) -> None:
    module, _, _, _ = models
    parent = tmp_path / "data/md2audio/models"
    parent.mkdir(parents=True)
    inherited = parent / ".test-melo.download-other"
    inherited.mkdir()
    witness = inherited / "keep"
    witness.write_bytes(b"owned by somebody else")

    class Cancelled(io.BytesIO):
        def read(self, size=-1):
            raise KeyboardInterrupt

        read1 = read

    monkeypatch.setattr(module, "build_opener", lambda *args: SimpleNamespace(open=lambda *args, **kwargs: Cancelled()))
    with pytest.raises(KeyboardInterrupt):
        module.ensure_model("melo")
    assert witness.read_bytes() == b"owned by somebody else"
    assert sorted(p.name for p in parent.glob(".test-melo.download-*")) == [inherited.name]
    assert not (parent / "test-melo").exists()


def test_corrupt_existing_model_is_preserved_without_network(models, tmp_path) -> None:
    module, _, _, downloads = models
    model = module.ensure_model("melo")
    file = model / "model.onnx"
    file.write_bytes(b"original altered model")
    with pytest.raises(RuntimeError, match="cached model"):
        module.ensure_model("melo")
    assert file.read_bytes() == b"original altered model"
    assert len(downloads) == 1


def test_concurrent_cold_starts_share_one_download(models, tmp_path, monkeypatch) -> None:
    module, _, payload, downloads = models
    started = threading.Event()
    release = threading.Event()
    results: list[Path] = []
    errors: list[BaseException] = []

    def download(url, timeout):
        downloads.append(url)
        started.set()
        if not release.wait(5):
            raise TimeoutError("fixture release missing")
        return io.BytesIO(payload)

    monkeypatch.setattr(module, "build_opener", lambda *args: SimpleNamespace(open=download))

    def worker():
        try:
            results.append(module.ensure_model("melo"))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    first.start()
    try:
        assert started.wait(5)
        second.start()
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors and len(downloads) == 1
    assert results == [tmp_path / "data/md2audio/models/test-melo"] * 2


def test_lock_wait_is_bounded(models, tmp_path, monkeypatch) -> None:
    import fcntl
    module, _, _, downloads = models
    parent = tmp_path / "data/md2audio/models"
    parent.mkdir(parents=True)
    lock = os.open(parent / ".test-melo.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX)
    monkeypatch.setattr(module, "_LOCK_TIMEOUT_SECONDS", 0.1)
    try:
        with pytest.raises(RuntimeError, match="lock"):
            module.ensure_model("melo")
    finally:
        os.close(lock)
    assert downloads == []


@pytest.mark.parametrize("ordering", ["before", "after"])
@pytest.mark.parametrize("command", ["synth", "bundle"])
def test_plain_local_flag_and_model_choice_route_without_manual_path(tmp_path, monkeypatch, ordering, command) -> None:
    source = tmp_path / "input.md"
    output = tmp_path / "result.mp3"
    model = tmp_path / "selected-model"
    source.write_text("**你好。**", encoding="utf-8")
    requests: list[str] = []
    calls: list[tuple] = []

    def choose(name):
        requests.append(name)
        return model

    def local(text, selected_model, speaker, speed, destination, volume):
        calls.append((text, selected_model, speaker, volume))
        destination.write_bytes(b"local mp3")
        return [{"text": text, "offset": 0, "duration": 10_000_000}]

    monkeypatch.setitem(sys.modules, "md2audio_models", SimpleNamespace(ensure_model=choose))
    monkeypatch.setitem(sys.modules, "md2audio_local", SimpleNamespace(synthesize_local=local, LocalTTSError=RuntimeError))
    monkeypatch.setattr(md2audio.shutil, "which", lambda name: "/installed/" + name)
    monkeypatch.setattr(md2audio, "synthesize", lambda *args: pytest.fail("Edge called in local mode"))
    monkeypatch.setattr(md2audio, "generate_video", lambda *args: None)
    options = ["--local-tts", "--local-model", "aishell3", "--voice", "10", "--volume", "1.50"]
    args = options + [str(source)] if ordering == "before" else [str(source)] + options
    monkeypatch.setattr(sys, "argv", ["md2audio", command, *args, "-o", str(output)])
    md2audio.main()
    assert requests == ["aishell3"]
    assert calls == [("你好。", model, "10", "1.50")]


def test_edge_and_help_do_not_import_model_module_or_create_xdg_state(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.md"
    source.write_text("hello")
    data = tmp_path / "data-is-a-file"
    data.write_bytes(b"must stay untouched")
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    monkeypatch.setitem(sys.modules, "md2audio_models", None)

    def edge(text, voice, speed, output):
        output.write_bytes(b"edge")
        return []

    monkeypatch.setattr(md2audio, "synthesize", edge)
    monkeypatch.setattr(sys, "argv", ["md2audio", str(source), "-o", str(tmp_path / "edge.mp3")])
    md2audio.main()
    monkeypatch.setattr(sys, "argv", ["md2audio", "--help"])
    with pytest.raises(SystemExit) as exc:
        md2audio.main()
    assert exc.value.code == 0
    assert data.read_bytes() == b"must stay untouched"


@pytest.mark.parametrize("options", [["--local-model", "melo"], ["--local-tts", "--voice", "1"],
                                    ["--local-tts", "--local-model", "aishell3", "--voice", "174"]])
def test_invalid_local_selection_rejected_before_input_or_model_io(monkeypatch, options) -> None:
    monkeypatch.setitem(sys.modules, "md2audio_models", None)
    monkeypatch.setattr(sys, "argv", ["md2audio", "missing.md", *options])
    with pytest.raises(SystemExit) as exc:
        md2audio.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("reason", ["empty", "nul", "dependency"])
def test_local_input_and_native_dependencies_validate_before_download(tmp_path, monkeypatch, reason) -> None:
    source = tmp_path / "input.md"
    source.write_text(" " if reason == "empty" else "nul\x00text" if reason == "nul" else "hello")
    output = tmp_path / "audio.mp3"
    output.write_bytes(b"original mp3")
    requests: list[str] = []
    monkeypatch.setitem(sys.modules, "md2audio_models", SimpleNamespace(ensure_model=lambda name: requests.append(name)))
    monkeypatch.setattr(md2audio.shutil, "which", lambda name: None if reason == "dependency" else "/installed/" + name)
    monkeypatch.setattr(sys, "argv", ["md2audio", str(source), "--local-tts", "-o", str(output)])
    with pytest.raises(SystemExit) as exc:
        md2audio.main()
    assert exc.value.code == 1
    assert requests == [] and output.read_bytes() == b"original mp3"


@pytest.mark.parametrize("kind", [tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE])
def test_archive_links_and_special_nodes_are_rejected(models, tmp_path, monkeypatch, kind) -> None:
    module, data, _, _ = models
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:bz2") as archive:
        for name, content in data.items():
            member = tarfile.TarInfo("test-melo/" + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        member = tarfile.TarInfo("test-melo/unsafe")
        member.type = kind
        member.linkname = "test-melo/tokens.txt"
        archive.addfile(member)
    payload = out.getvalue()
    old = module._MODELS["melo"]
    monkeypatch.setitem(module._MODELS, "melo", (old[0], len(payload), hashlib.sha256(payload).hexdigest(), *old[3:]))
    monkeypatch.setattr(module, "build_opener", lambda *args: SimpleNamespace(open=lambda *args, **kwargs: io.BytesIO(payload)))
    with pytest.raises(RuntimeError, match="archive"):
        module.ensure_model("melo")
    assert not (tmp_path / "data/md2audio/models/test-melo").exists()


def test_download_has_an_overall_deadline(models, tmp_path, monkeypatch) -> None:
    module, _, payload, _ = models
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])

    class Slow(io.BytesIO):
        def read(self, size=-1):
            now[0] = 700.0
            return super().read(size)

        read1 = read

    monkeypatch.setattr(module, "build_opener", lambda *args: SimpleNamespace(open=lambda *args, **kwargs: Slow(payload)))
    with pytest.raises(RuntimeError, match="timed out"):
        module.ensure_model("melo")
    assert not (tmp_path / "data/md2audio/models/test-melo").exists()


def test_world_writable_data_ancestor_is_refused_without_network(models, tmp_path, monkeypatch) -> None:
    module, _, _, downloads = models
    directory = tmp_path / "untrusted"
    directory.mkdir(mode=0o777)
    directory.chmod(0o777)
    monkeypatch.setenv("XDG_DATA_HOME", str(directory / "data"))
    with pytest.raises(RuntimeError, match="Unsafe model directory"):
        module.ensure_model("melo")
    assert not (directory / "data").exists() and downloads == []


def test_hidden_untrusted_symlink_hop_is_refused_without_external_changes(models, tmp_path, monkeypatch) -> None:
    module, _, _, downloads = models
    unsafe = tmp_path / "unsafe-hop"
    unsafe.mkdir()
    unsafe.chmod(0o777)
    external = tmp_path / "external"
    external.mkdir()
    (unsafe / "next").symlink_to(external)
    link = tmp_path / "configured-data"
    link.symlink_to(unsafe / "next")
    monkeypatch.setenv("XDG_DATA_HOME", str(link))
    with pytest.raises(RuntimeError, match="Unsafe model directory"):
        module.ensure_model("melo")
    assert list(external.iterdir()) == [] and link.is_symlink() and downloads == []


def test_foreign_owned_ancestor_is_refused_before_network(models, tmp_path, monkeypatch) -> None:
    module, _, _, downloads = models
    directory = tmp_path / "foreign"
    directory.mkdir()
    original = Path.lstat

    def stat(path):
        result = original(path)
        if path == directory:
            fields = list(result)
            fields[4] = 12345
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(Path, "lstat", stat)
    monkeypatch.setenv("XDG_DATA_HOME", str(directory / "data"))
    with pytest.raises(RuntimeError, match="Unsafe model directory"):
        module.ensure_model("melo")
    assert not (directory / "data").exists() and downloads == []


def test_permission_failure_has_no_alternate_cache_directory(models, tmp_path, monkeypatch) -> None:
    module, _, _, downloads = models
    original = Path.mkdir
    data = tmp_path / "data"

    def denied(path, *args, **kwargs):
        if path == data:
            raise PermissionError("fixture EACCES")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", denied)
    with pytest.raises(RuntimeError, match="EACCES"):
        module.ensure_model("melo")
    assert not data.exists() and downloads == []


def test_model_download_failure_keeps_original_cli_output(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.md"
    source.write_text("hello")
    output = tmp_path / "audio.mp3"
    output.write_bytes(b"last good mp3")

    def unavailable(name):
        raise RuntimeError("model download unavailable")

    monkeypatch.setitem(sys.modules, "md2audio_models", SimpleNamespace(ensure_model=unavailable))
    monkeypatch.setattr(md2audio.shutil, "which", lambda name: "/installed/" + name)
    monkeypatch.setattr(sys, "argv", ["md2audio", "--local-tts", str(source), "-o", str(output)])
    with pytest.raises(SystemExit) as exc:
        md2audio.main()
    assert exc.value.code == 1 and output.read_bytes() == b"last good mp3"


def test_simultaneous_creation_of_xdg_parents_succeeds(models, monkeypatch) -> None:
    module, _, _, downloads = models
    data = Path(os.environ["XDG_DATA_HOME"])
    rendezvous = threading.Barrier(2, timeout=5)
    original = Path.mkdir
    errors: list[BaseException] = []
    results: list[Path] = []

    def mkdir(path, *args, **kwargs):
        if path == data:
            rendezvous.wait()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)

    def worker():
        try:
            results.append(module.ensure_model("melo"))
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=worker), threading.Thread(target=worker)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(5)
    assert not any(worker.is_alive() for worker in workers)
    assert not errors and len(results) == 2 and len(downloads) == 1


def test_nonregular_cached_file_is_rejected_without_blocking(models, monkeypatch) -> None:
    import json
    module, _, _, _ = models
    directory = module.ensure_model("melo")
    tokens = directory / "tokens.txt"
    tokens.unlink()
    os.mkfifo(tokens, mode=0o600)
    script = """
import json, signal, sys
import md2audio_models
signal.alarm(2)
md2audio_models._MODELS = json.loads(sys.argv[1])
try:
    md2audio_models.ensure_model('melo')
except RuntimeError as exc:
    assert 'cached model' in str(exc), str(exc)
else:
    raise AssertionError('FIFO accepted')
"""
    result = subprocess.run([sys.executable, "-c", script, json.dumps(module._MODELS)],
                            capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_cached_group_writable_model_is_not_used(models) -> None:
    module, _, _, downloads = models
    directory = module.ensure_model("melo")
    (directory / "model.onnx").chmod(0o666)
    with pytest.raises(RuntimeError, match="cached model"):
        module.ensure_model("melo")
    assert len(downloads) == 1


def test_dangling_directory_symlink_is_not_retargeted_or_created(models, tmp_path, monkeypatch) -> None:
    module, _, _, downloads = models
    target = tmp_path / "missing-target"
    link = tmp_path / "configured-link"
    link.symlink_to(target)
    monkeypatch.setenv("XDG_DATA_HOME", str(link))
    with pytest.raises(RuntimeError):
        module.ensure_model("melo")
    assert link.is_symlink() and not target.exists() and not downloads


@pytest.mark.parametrize("platform,private_boundary,system_gid,valid", [
    ("android", True, 1000, True), ("linux", True, 1000, False),
    ("android", False, 1000, False), ("android", True, 12345, False),
])
def test_termux_system_ancestry_is_narrowly_admitted(models, monkeypatch, platform, private_boundary, system_gid, valid) -> None:
    import stat
    module, _, _, _ = models
    path = Path("/data/data/com.termux/files/home/.local/share")
    system = {Path("/data"), Path("/data/data")}
    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setattr(module.os, "getuid", lambda: 10175)
    monkeypatch.setattr(module.os, "getgid", lambda: 10175)

    def metadata(candidate):
        if candidate in system:
            owner, group, mode = 1000, system_gid, 0o771
        elif candidate == Path("/data/data/com.termux"):
            owner, group, mode = 10175, 10175, 0o700 if private_boundary else 0o711
        elif candidate == Path("/data/data/com.termux/files"):
            owner, group, mode = 10175, 10175, 0o771
        else:
            owner, group, mode = 10175, 10175, 0o700
        return os.stat_result((stat.S_IFDIR | mode, 1, 1, 1, owner, group, 0, 0, 0, 0))

    monkeypatch.setattr(Path, "lstat", metadata)
    if valid:
        assert module._safe_directory(path) == path
    else:
        with pytest.raises(RuntimeError, match="Unsafe model directory"):
            module._safe_directory(path)


def test_downloader_reads_available_bytes_without_waiting_to_fill_large_buffer(models, monkeypatch) -> None:
    module, _, payload, _ = models

    class Available(io.BytesIO):
        def read(self, size=-1):
            raise AssertionError("read() can wait indefinitely for a large buffer under a trickling response")

        def read1(self, size=-1):
            return super().read(min(size, 32))

    monkeypatch.setattr(module, "build_opener", lambda *args: SimpleNamespace(open=lambda *args, **kwargs: Available(payload)))
    assert module.ensure_model("melo").is_dir()


def test_http_protocol_failure_is_an_actionable_model_error(models, tmp_path, monkeypatch) -> None:
    import http.client
    module, _, _, _ = models

    class Broken(io.BytesIO):
        def read(self, size=-1):
            raise http.client.IncompleteRead(b"partial", 100)

        read1 = read

    monkeypatch.setattr(module, "build_opener", lambda *args: SimpleNamespace(open=lambda *args, **kwargs: Broken()))
    with pytest.raises(RuntimeError, match="Could not prepare local model"):
        module.ensure_model("melo")
    assert not (tmp_path / "data/md2audio/models/test-melo").exists()


def test_model_import_does_not_create_or_read_xdg_data(tmp_path, monkeypatch) -> None:
    data = tmp_path / "xdg-is-a-file"
    data.write_bytes(b"untouched")
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    result = subprocess.run([sys.executable, "-c", "import md2audio_models; import md2audio"],
                            capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert data.read_bytes() == b"untouched"
