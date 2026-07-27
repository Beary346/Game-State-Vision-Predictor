from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"


def find_test_screenshots() -> list[Path]:
    """Return all PNG files under ``data/bronze/``, sorted."""
    return sorted(BRONZE_DIR.glob("*.png"))


def find_first_screenshot() -> Path | None:
    """Return the first available test screenshot, or ``None``."""
    screenshots = find_test_screenshots()
    return screenshots[0] if screenshots else None


@pytest.fixture(scope="session")
def bronze_dir() -> Path:
    return BRONZE_DIR


@pytest.fixture(scope="session")
def test_screenshot() -> Path:
    """Provide the first available PNG in ``data/bronze/``."""
    screenshots = find_test_screenshots()
    if not screenshots:
        pytest.skip("No test screenshots found in data/bronze/")
    return screenshots[0]


@pytest.fixture(scope="session")
def test_screenshot_path(test_screenshot: Path) -> str:
    return str(test_screenshot)
