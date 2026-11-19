"""Password hashing and account loading for the web control panel.

Passwords are stored as salted PBKDF2-SHA256 hashes (Python standard library,
no extra dependency). Generate a hash with:

    python scripts/hash_password.py

and put it in .env as e.g. ADMIN_PASSWORD_HASH=pbkdf2_sha256$...
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
from typing import Dict, Optional

logger = logging.getLogger("YarnBreak.auth")

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 310_000  # OWASP recommendation for PBKDF2-SHA256


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    """Return 'pbkdf2_sha256$<iterations>$<salt>$<hash>' (base64 parts)."""
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    b64 = lambda b: base64.b64encode(b).decode("ascii")
    return f"{ALGORITHM}${iterations}${b64(salt)}${b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of a password against a stored hash."""
    try:
        algorithm, iterations, salt_b64, hash_b64 = encoded.split("$")
        if algorithm != ALGORITHM:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected)


def _password_hash_from_env(prefix: str) -> Optional[str]:
    """Prefer <PREFIX>_PASSWORD_HASH. A plain <PREFIX>_PASSWORD is still accepted
    (hashed in memory) so old .env files keep working, but logs a warning."""
    hashed = os.getenv(f"{prefix}_PASSWORD_HASH")
    if hashed:
        return hashed
    plain = os.getenv(f"{prefix}_PASSWORD")
    if plain:
        logger.warning("%s_PASSWORD is stored in plain text; use %s_PASSWORD_HASH "
                       "(python scripts/hash_password.py)", prefix, prefix)
        return hash_password(plain)
    return None


# (env prefix, default username, role)
ACCOUNT_SPECS = (
    ("ADMIN", "admin", "admin"),
    ("USER1", "user1", "user"),
    ("USER2", "user2", "user"),
)


def load_accounts() -> Dict[str, dict]:
    """username -> {'password_hash', 'role'}. Accounts without a password in
    the environment are disabled: there is no built-in default password."""
    accounts: Dict[str, dict] = {}
    for prefix, default_user, role in ACCOUNT_SPECS:
        pw_hash = _password_hash_from_env(prefix)
        if not pw_hash:
            logger.warning("Account %s disabled: no %s_PASSWORD_HASH set", prefix, prefix)
            continue
        username = os.getenv(f"{prefix}_USERNAME", default_user)
        accounts[username] = {"password_hash": pw_hash, "role": role}
    return accounts
