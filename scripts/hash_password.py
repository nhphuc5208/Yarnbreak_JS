#!/usr/bin/env python3
"""Print a PBKDF2 hash to paste into .env (e.g. ADMIN_PASSWORD_HASH=...)."""
import getpass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yarnbreak.auth import hash_password  # noqa: E402


def main() -> int:
    pw = getpass.getpass("Password: ")
    if pw != getpass.getpass("Repeat:   "):
        print("Passwords do not match.", file=sys.stderr)
        return 1
    print(hash_password(pw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
