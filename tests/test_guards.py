import pytest

from k8s_aiops_agent.guards import (
    REDACTED,
    ToolForbiddenError,
    ensure_namespace_allowed,
    redact,
    truncate,
)


def test_ensure_namespace_allowed_with_listed_namespace_passes() -> None:
    ensure_namespace_allowed("scenario-01", ["scenario-01"])


def test_ensure_namespace_allowed_outside_allowlist_raises_forbidden() -> None:
    with pytest.raises(ToolForbiddenError, match="kube-system"):
        ensure_namespace_allowed("kube-system", ["scenario-01"])


@pytest.mark.parametrize(
    "text",
    [
        '{"DB_PASSWORD": "hunter2"}',
        "api_key=hunter2",
        "Authorization: Bearer hunter2",
        "token: eyJhbGciOi.eyJzdWIiOi.c2lnbmF0dXJl",
    ],
)
def test_redact_secret_like_values_removes_the_value(text: str) -> None:
    redacted = redact(text)

    assert "hunter2" not in redacted
    assert "eyJ" not in redacted
    assert REDACTED in redacted


def test_redact_ordinary_config_keeps_text_unchanged() -> None:
    text = '{"PAYMENT_TIMEOUT_MS": "30s", "replicas": 2}'

    assert redact(text) == text


def test_truncate_short_text_returns_it_unchanged() -> None:
    assert truncate("short", 100) == "short"


def test_truncate_long_text_keeps_head_and_tail_within_limit() -> None:
    text = "START" + "x" * 1_000 + "FATAL"

    result = truncate(text, 100)

    assert result.startswith("START")
    assert result.endswith("FATAL")
    assert "[truncated 910 chars]" in result
