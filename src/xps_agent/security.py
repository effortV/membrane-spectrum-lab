from __future__ import annotations

import hashlib
import hmac
import os
import re
import subprocess
from pathlib import Path


def safe_error(error: object, *, limit: int = 2000) -> str:
    """Scrub stored credentials and common credential URL/header formats."""
    text = str(error)
    for name, value in os.environ.items():
        if value and any(token in name.upper() for token in ("API_KEY", "INSTTOKEN", "PASSWORD")):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(
        r"(?i)(api[_-]?key|access[_-]?token|insttoken|authorization)(\s*[=:]\s*)[^\s&,;]+",
        r"\1\2[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", text)
    return text[:limit]


def secure_local_file(path: Path) -> None:
    if os.name != "nt":
        path.chmod(0o600)
        return
    identity = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True, check=True
    )
    match = re.search(r"S-1-[0-9-]+", identity.stdout)
    if not match:
        raise RuntimeError("Cannot resolve current Windows user for credential file permissions")
    result = subprocess.run(
        [
            "icacls",
            str(path),
            "/inheritance:r",
            "/grant:r",
            f"*{match.group(0)}:(F)",
            "*S-1-5-18:(F)",
            "*S-1-5-32-544:(F)",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "Cannot restrict credential file permissions; configuration was not saved"
        )


def hash_ui_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("界面访问密码至少需要 12 个字符。")
    salt = os.urandom(16)
    iterations = 310000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_ui_password(password: str, stored: str) -> bool:
    try:
        algorithm, rounds, salt, expected = stored.split("$")
        iterations = int(rounds)
        if algorithm != "pbkdf2_sha256" or not 100000 <= iterations <= 1000000:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), iterations)
        return hmac.compare_digest(actual.hex(), expected)
    except (ValueError, TypeError):
        return False
