"""The tracked push hook preserves stdin/arguments and fails closed without Lefthook."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


HOOK = Path(__file__).resolve().parents[1] / "scripts/hooks/pre-push"


def test_missing_lefthook_blocks_push(tmp_path):
    env = {**os.environ, "PATH": str(tmp_path)}
    env.pop("LEFTHOOK_BIN", None)
    result = subprocess.run(["/bin/bash", str(HOOK)], env=env, capture_output=True, timeout=5)
    assert result.returncode != 0
    assert b"lefthook" in result.stderr


@pytest.mark.parametrize("variable,value", [
    ("LEFTHOOK", "0"),
    ("LEFTHOOK", "false"),
    ("LEFTHOOK_CONFIG", "missing-or-alternate.yml"),
    ("LEFTHOOK_EXCLUDE", "branch-protection,review-check"),
])
def test_execution_overrides_cannot_skip_gates(tmp_path, variable, value):
    executor = tmp_path / "lefthook"
    executor.write_text("#!/bin/sh\nexit 0\n")
    executor.chmod(0o700)
    env = {**os.environ, "PATH": str(tmp_path)}
    for name in ("LEFTHOOK", "LEFTHOOK_BIN", "LEFTHOOK_CONFIG", "LEFTHOOK_EXCLUDE"):
        env.pop(name, None)
    env[variable] = value
    result = subprocess.run(["/bin/bash", str(HOOK)], env=env, capture_output=True, timeout=5)
    assert result.returncode != 0
    assert b"LEFTHOOK" in result.stderr


@pytest.mark.skipif(not shutil.which("lefthook"), reason="Real Lefthook integration")
def test_real_lefthook_runs_required_checks():
    repo = HOOK.parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    env = dict(os.environ)
    for name in ("LEFTHOOK", "LEFTHOOK_BIN", "LEFTHOOK_CONFIG", "LEFTHOOK_EXCLUDE", "LEFTHOOK_OUTPUT"):
        env.pop(name, None)
    result = subprocess.run(["/bin/bash", str(HOOK), "origin", "safe-url"],
                            input=f"refs/heads/topic {head} refs/heads/main {'0' * 40}\n",
                            cwd=repo, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "branch-protection" in result.stdout
    assert "review-check" in result.stdout
    assert "direct pushes to protected branches are forbidden" in result.stderr + result.stdout


def test_lefthook_receives_push_stdin_arguments_and_controls_exit(tmp_path):
    executor = tmp_path / "lefthook"
    executor.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$HOOK_ARGV"\n/bin/cat > "$HOOK_STDIN"\nexit 37\n')
    executor.chmod(0o700)
    argv, stdin = tmp_path / "argv", tmp_path / "stdin"
    env = {**os.environ, "PATH": str(tmp_path), "HOOK_ARGV": str(argv), "HOOK_STDIN": str(stdin)}
    env.pop("LEFTHOOK_BIN", None)
    payload = b"refs/heads/topic aaaa refs/heads/topic bbbb\n"
    result = subprocess.run(["/bin/bash", str(HOOK), "origin", "safe-url"],
                            input=payload, env=env, capture_output=True, timeout=5)
    assert result.returncode == 37
    assert argv.read_text().splitlines() == ["run", "pre-push", "origin", "safe-url"]
    assert stdin.read_bytes() == payload
