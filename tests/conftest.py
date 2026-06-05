# SPDX-License-Identifier: MIT
"""Test scaffold for GLOSSATE."""

import pytest


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK


@pytest.fixture(autouse=True)
def _isolate_transcript_cache(monkeypatch, tmp_path):
    """Disable the transcript cache by default and keep it off the real home dir.

    Dedicated cache tests re-enable it by clearing GLOSSATE_NO_CACHE.
    """
    monkeypatch.setenv("GLOSSATE_NO_CACHE", "1")
    monkeypatch.setenv("GLOSSATE_CACHE_DIR", str(tmp_path / "glossate-cache"))
