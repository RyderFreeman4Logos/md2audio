#!/usr/bin/env bash
# Validate native review and local-gate evidence for the exact commit being pushed.
set -euo pipefail

python3 /dev/fd/3 3<<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


try:
    head = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    updates = [line.split() for line in sys.stdin.read().splitlines()]
    if not updates or any(len(update) != 4 or update[1] != head for update in updates):
        raise ValueError("push must contain only the reviewed HEAD")
    if any(update[2] in {"refs/heads/main", "refs/heads/master", "refs/heads/dev"} for update in updates):
        raise ValueError("direct pushes to protected branches are forbidden")
    for args in (("diff", "--quiet", "--"), ("diff", "--cached", "--quiet", "--")):
        if subprocess.run(["git", *args]).returncode:
            raise ValueError("tracked changes are not committed")
    receipt_path = Path.home() / "tmp/md2audio-reviews" / f"{head}.json"
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "head": head, "tree": tree, "verdict": "PASS",
        "gate_tree": tree, "gate_command": "just check", "gate_exit_code": 0,
    }
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("review or gate receipt does not match HEAD")
    if not isinstance(receipt.get("reviewer"), str) or not receipt["reviewer"].strip():
        raise ValueError("native reviewer identity is missing")
    if type(receipt["gate_exit_code"]) is not int:
        raise ValueError("gate exit code must be an integer")
    # shortcut: receipts trust local agents; authenticate storage if writers become untrusted.
    artifacts = {
        artifact: Path(receipt[f"{artifact}_path"]).read_bytes()
        for artifact in ("report", "gate_log")
    }
    for artifact, evidence in artifacts.items():
        digest = hashlib.sha256(evidence).hexdigest()
        if digest != receipt[f"{artifact}_sha256"]:
            raise ValueError(f"{artifact} evidence digest does not match")
    report = json.loads(artifacts["report"])
    base = git("merge-base", "origin/main", "HEAD")
    diff = subprocess.check_output(["git", "diff", "--binary", "origin/main...HEAD"])
    report_expected = {
        "head": head, "tree": tree, "base": base, "verdict": "PASS", "findings": [],
        "diff_sha256": hashlib.sha256(diff).hexdigest(), "reviewer": receipt["reviewer"],
    }
    if not isinstance(report, dict) or any(report.get(k) != v for k, v in report_expected.items()):
        raise ValueError("native report verdict, findings, or candidate identity does not match")
    gate_log = artifacts["gate_log"].decode("utf-8")
    if not gate_log.startswith(f"GATE_TREE={tree}\nGATE_COMMAND=just check\n"):
        raise ValueError("gate log does not identify this tree and command")
    if not gate_log.endswith("\nGATE_EXIT_CODE=0\n"):
        raise ValueError("gate log does not record a successful exit")
except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
    print(f"Push blocked: native review/local gate evidence is missing or invalid: {exc}", file=sys.stderr)
    sys.exit(1)

print(f"pre-push: Native review and local gate verified for HEAD {head[:11]}.")
PY
