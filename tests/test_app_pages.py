"""Smoke test: every page renders against the saved results without an exception."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted((ROOT / "app" / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_page_renders(page):
    at = AppTest.from_file(str(page), default_timeout=180)
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def test_home_navigation():
    at = AppTest.from_file(str(ROOT / "app" / "Home.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
