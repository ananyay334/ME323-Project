"""Shared fixtures: a synthetic dataset generated into a temporary folder."""
from pathlib import Path

import pytest

from ml.synthetic import generate
from ml.tests._util import synth_config


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synthetic")
    generate(out, seed=323, verbose=False)
    return out


@pytest.fixture(scope="session")
def cfg(synth_dir) -> dict:
    return synth_config(synth_dir)


@pytest.fixture(scope="session")
def data(cfg):
    from ml.data import load_data
    return load_data(cfg, "synthetic")


def pytest_configure(config):
    # loky's fork+exec of worker processes trips this Python >= 3.12 notice; it is harmless
    # because the child immediately execs a fresh interpreter.
    config.addinivalue_line(
        "filterwarnings", "ignore:This process .* is multi-threaded.*:DeprecationWarning")
