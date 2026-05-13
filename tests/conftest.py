import pytest
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_data: test requires the data/ directory to be present"
    )


@pytest.fixture(scope="session")
def data_dir():
    if not DATA_DIR.is_dir():
        pytest.skip("data/ directory not found — download it first")
    return DATA_DIR
