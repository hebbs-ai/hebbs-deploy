"""Tests for the exception hierarchy."""

from __future__ import annotations

from hebbs.exceptions import (
    HebbsError,
    HebbsConnectionError,
    HebbsTimeoutError,
    HebbsNotFoundError,
    HebbsUnavailableError,
    HebbsInvalidArgumentError,
    HebbsInternalError,
)


def test_exception_hierarchy():
    assert issubclass(HebbsConnectionError, HebbsError)
    assert issubclass(HebbsTimeoutError, HebbsError)
    assert issubclass(HebbsNotFoundError, HebbsError)
    assert issubclass(HebbsUnavailableError, HebbsError)
    assert issubclass(HebbsInvalidArgumentError, HebbsError)
    assert issubclass(HebbsInternalError, HebbsError)


def test_exception_message():
    e = HebbsConnectionError("connection refused")
    assert "connection refused" in str(e)
