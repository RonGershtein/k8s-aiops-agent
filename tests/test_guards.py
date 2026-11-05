import pytest

from k8s_aiops_agent.guards import (
    REDACTED,
    ToolForbiddenError,
    ensure_namespace_allowed,
    redact,
    truncate,
)


def test_listed_namespace_is_allowed() -> None:
    ensure_namespace_allowed("scenario-01", ["scenario-01"])


def test_namespace_outside_allowlist_is_forbidden() -> None:
    with pytest.raises(ToolForbiddenError, match="kube-system"):
        ensure_namespace_allowed("kube-system", ["scenario-01"])


@pytest.mark.parametrize(
    "text",
    [
        '{"DB_PASSWORD": "hunter2"}',
        '{"env": [{"name": "DB_PASSWORD", "value": "hunter2"}]}',
        "api_key=hunter2",
        "Authorization: Bearer hunter2",
        "token: eyJhbGciOi.eyJzdWIiOi.c2lnbmF0dXJl",
        '{"db_password": "correct hunter2 battery"}',
        '{"api_key": "abc\\"hunter2"}',
        "Authorization: Basic hunter2==",
        '{"DATABASE_URL": "postgres://admin:hunter2@db:5432/app"}',
        # a YAML/JSON file inside a ConfigMap value: the quotes arrive JSON-escaped
        r'{"config.yaml": "db:\n  password: \"correct hunter2\"\n"}',
        r'{"app.json": "{\"api_key\": \"correct hunter2\"}"}',
        "postgres://admin:pa/hunter2@db:5432",
        "API_KEY='correct hunter2'",
    ],
)
def test_redact_secret_like_values_removes_the_value(text: str) -> None:
    redacted = redact(text)

    assert "hunter2" not in redacted
    assert "eyJ" not in redacted
    assert REDACTED in redacted


@pytest.mark.parametrize(
    "text",
    [
        '{"PAYMENT_TIMEOUT_MS": "30s", "replicas": 2}',
        '{"valueFrom": {"secretKeyRef": {"name": "db", "key": "password"}}}',
        '{"secretName": "db-creds", "tokenFile": "/var/run/token"}',
    ],
)
def test_redact_ordinary_config_keeps_text_unchanged(text: str) -> None:
    assert redact(text) == text


def test_truncate_short_text_returns_it_unchanged() -> None:
    assert truncate("short", 100) == "short"


def test_truncate_keeps_head_and_tail() -> None:
    text = "START" + "x" * 1_000 + "FATAL"

    result = truncate(text, 100)

    assert result.startswith("START")
    assert result.endswith("FATAL")
    assert "[truncated 910 chars]" in result
