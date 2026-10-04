import pytest

from sciai.config import DEFAULT_FILE, load


def test_default_file_loads():
    cfg = load([DEFAULT_FILE])
    assert cfg.model.num_ctx == 8192 and cfg.risk.stakes_threshold == 2


def test_unknown_keys_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[model]\nnum_ctxx = 1\n")
    with pytest.raises(ValueError):
        load([p])
