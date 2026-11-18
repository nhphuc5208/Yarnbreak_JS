"""Password hashing and account loading for the web panel."""
import pytest

import auth


def test_hash_round_trip():
    h = auth.hash_password("s3cret", iterations=1000)
    assert h.startswith("pbkdf2_sha256$1000$")
    assert auth.verify_password("s3cret", h)
    assert not auth.verify_password("wrong", h)


def test_same_password_gets_different_salt():
    assert auth.hash_password("x", iterations=1000) != auth.hash_password("x", iterations=1000)


@pytest.mark.parametrize("bad", ["", "plain-text", "md5$1$a$b", "pbkdf2_sha256$x$y$z"])
def test_verify_rejects_malformed_hash(bad):
    assert not auth.verify_password("anything", bad)


def test_empty_password_cannot_be_hashed():
    with pytest.raises(ValueError):
        auth.hash_password("")


@pytest.fixture
def clean_env(monkeypatch):
    for prefix, _, _ in auth.ACCOUNT_SPECS:
        for suffix in ("USERNAME", "PASSWORD", "PASSWORD_HASH"):
            monkeypatch.delenv(f"{prefix}_{suffix}", raising=False)
    return monkeypatch


def test_no_default_passwords(clean_env):
    assert auth.load_accounts() == {}


def test_accounts_from_hash_and_legacy_plaintext(clean_env):
    clean_env.setenv("ADMIN_USERNAME", "boss")
    clean_env.setenv("ADMIN_PASSWORD_HASH", auth.hash_password("admin-pw", iterations=1000))
    clean_env.setenv("USER1_PASSWORD", "user-pw")  # legacy plain text still works

    accounts = auth.load_accounts()
    assert set(accounts) == {"boss", "user1"}
    assert accounts["boss"]["role"] == "admin"
    assert auth.verify_password("admin-pw", accounts["boss"]["password_hash"])
    assert auth.verify_password("user-pw", accounts["user1"]["password_hash"])
