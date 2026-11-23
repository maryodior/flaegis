"""Shared setup for the FLAegis tests. Every test gets a fresh, temporary database,
so the real flaegis.db is never touched."""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: runs a real federated training session (several minutes)')


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv('FLAEGIS_DB', str(tmp_path / 'test.db'))
    import database
    importlib.reload(database)
    database.initialise()
    yield database


@pytest.fixture
def model():
    import model as m
    return m
