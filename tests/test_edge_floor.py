"""The declared Edge floor is what install resolution is allowed to use."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_edge_tts_floor_rejects_403_client_and_allows_later_stable() -> None:
    dependencies = (ROOT / "pyproject.toml").read_text().splitlines()
    assert '    "edge-tts>=7.3.0",' in dependencies
    assert '    "edge-tts>=7.0",' not in dependencies
    readme = (ROOT / "README.md").read_text()
    assert 'uv tool install --force --upgrade "md2audio @ git+' in readme
    assert "does not update by itself" in readme
