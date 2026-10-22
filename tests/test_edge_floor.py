"""The declared Edge floor is what install resolution is allowed to use."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_edge_tts_floor_rejects_403_client_and_allows_later_stable() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["dependencies"] == [
        "edge-tts>=7.3.0",
        "tomli>=2.0; python_version < '3.11'",
    ]
    readme = (ROOT / "README.md").read_text()
    assert 'uv tool install --force --upgrade "md2audio @ git+' in readme
    assert "does not update by itself" in readme
