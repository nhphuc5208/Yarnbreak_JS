"""i18n consistency for the web UI (static/i18n.js).

A JavaScript object literal silently keeps the LAST value of a duplicated key,
so duplicates are invisible at runtime; this check reads the source text.
tests/check_i18n.js additionally verifies that every key used in the HTML/JS
has a translation (run by CI with Node).
"""
import re
from pathlib import Path

import pytest

SRC = (Path(__file__).resolve().parent.parent / "static" / "i18n.js").read_text(encoding="utf-8")
LANGS = ("vi", "ko", "en")
KEY = re.compile(r'^\s*"([^"]+)"\s*:', re.MULTILINE)


def _blocks():
    """Split `const I18N = { vi: {...}, ko: {...}, en: {...} };` into per-language text."""
    body = SRC[SRC.index("const I18N = {"):]
    body = body[:body.index("\n};")]
    starts = [(lang, re.search(rf"^\s{{4}}{lang}\s*:\s*\{{", body, re.MULTILINE)) for lang in LANGS]
    assert all(m for _, m in starts), "missing a language block"
    starts.sort(key=lambda x: x[1].start())
    out = {}
    for i, (lang, m) in enumerate(starts):
        end = starts[i + 1][1].start() if i + 1 < len(starts) else len(body)
        out[lang] = body[m.end():end]
    return out


BLOCKS = _blocks()


@pytest.mark.parametrize("lang", LANGS)
def test_no_duplicate_keys(lang):
    keys = KEY.findall(BLOCKS[lang])
    dups = sorted({k for k in keys if keys.count(k) > 1})
    assert not dups, f"duplicate keys in '{lang}': {dups}"


@pytest.mark.parametrize("lang", LANGS[1:])
def test_same_key_set_as_vietnamese(lang):
    vi = set(KEY.findall(BLOCKS["vi"]))
    other = set(KEY.findall(BLOCKS[lang]))
    assert other == vi, (f"missing in {lang}: {sorted(vi - other)}; "
                         f"extra in {lang}: {sorted(other - vi)}")


def test_no_empty_translations():
    empty = re.compile(r'^\s*"([^"]+)"\s*:\s*""\s*,?\s*$', re.MULTILINE)
    found = {lang: empty.findall(text) for lang, text in BLOCKS.items()}
    assert not any(found.values()), found
