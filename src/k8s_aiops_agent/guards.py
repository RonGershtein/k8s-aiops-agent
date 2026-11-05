import re
from collections.abc import Collection


class ToolForbiddenError(Exception):
    """The agent asked for something outside its allowed boundary."""


_SECRET_WORD = r"(?:password|passwd|secret|token|api[_-]?key|private[_-]?key|credentials?)"
# Keys like secretKeyRef, secretName or tokenFile point at a secret; their values are not secret.
_SECRET_KEY = rf"[\w.-]*{_SECRET_WORD}[\w.-]*(?<!ref)(?<!name)(?<!file)(?<!path)(?<!dir)(?<!mount)"
_KEY_END = r'(?:\\?")?'
# Redacts up to the closing quote: "...", '...', or \"...\" inside a JSON string.
_SECRET_QUOTED_VALUE = re.compile(
    rf"""(?i)({_SECRET_KEY}{_KEY_END}\s*[:=]\s*(\\?["']))((?:(?!\2)(?:[^\\]|\\.))*)"""
)
_SECRET_KEY_VALUE = re.compile(
    rf"""(?i)({_SECRET_KEY}{_KEY_END}\s*[:=]\s*)([^"'\s,}}{{\[\\][^"\s,}}\\]*)"""
)
_BASIC_AUTH = re.compile(r"(?i)(authorization\"?\s*[:=]\s*\"?basic\s+)[\w+/]+=*")
_URL_PASSWORD = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^/\s:@\"]+:)[^@\s\"]+?(?=@)")
# Kubernetes env vars serialize as {"name": "DB_PASSWORD", "value": "..."}: the name is the key.
_SECRET_ENV_VAR = re.compile(rf'(?i)("name":\s*"{_SECRET_KEY}",\s*"value":\s*")((?:[^"\\]|\\.)*)')
_BEARER = re.compile(r"(?i)(bearer\s+)[\w.~+/-]+=*")
_JWT = re.compile(r"eyJ[\w-]+\.[\w-]+\.[\w-]+")

REDACTED = "[REDACTED]"


_ESCAPES_AND_SPACE = re.compile(r'[\\"\s]+')


def normalize_text(text: str) -> str:
    """Ignores JSON escaping and whitespace, so quotes copied from JSON output still match."""
    return _ESCAPES_AND_SPACE.sub(" ", text).strip().lower()


def ensure_namespace_allowed(namespace: str, allowed: Collection[str]) -> None:
    if namespace not in allowed:
        raise ToolForbiddenError(
            f"namespace '{namespace}' is not allowed; allowed: {', '.join(sorted(allowed))}"
        )


def redact(text: str) -> str:
    text = _JWT.sub(REDACTED, text)
    text = _BEARER.sub(rf"\g<1>{REDACTED}", text)
    text = _BASIC_AUTH.sub(rf"\g<1>{REDACTED}", text)
    text = _URL_PASSWORD.sub(rf"\g<1>{REDACTED}", text)
    text = _SECRET_ENV_VAR.sub(rf"\g<1>{REDACTED}", text)
    text = _SECRET_QUOTED_VALUE.sub(rf"\g<1>{REDACTED}", text)
    return _SECRET_KEY_VALUE.sub(rf"\g<1>{REDACTED}", text)


def truncate(text: str, limit: int) -> str:
    """Keeps the head and the tail, because errors usually sit at the end of logs."""
    if len(text) <= limit:
        return text
    half = limit // 2
    dropped = len(text) - 2 * half
    return f"{text[:half]}\n...[truncated {dropped} chars]...\n{text[-half:]}"
