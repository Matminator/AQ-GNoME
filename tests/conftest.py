import numpy as np
import pandas as pd
import pytest
import h5py
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ARRAY_SHAPE = (31, 19)  # U-grid × pH-grid


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_data: test requires the data/ directory to be present"
    )
    config.addinivalue_line(
        "markers", "requires_api: test requires MP_API_KEY and live network access"
    )


@pytest.fixture(scope="session")
def data_dir():
    if not DATA_DIR.is_dir():
        pytest.skip("data/ directory not found — download it first")
    return DATA_DIR


@pytest.fixture
def synthetic_df():
    """Minimal DataFrame matching Data_Handler schema — no real files needed.

    Four materials: m001/m002 are Fe-O oxides, m003 is Ti-O, m004 is Ni-O.
    mixed_pbx_save_id for m002 is 'Not computed' to test GGA fallback.
    Decomposition energies in h5_path_with_data are 0.1*(row+1):
        m001 → all 0.1, m002 → all 0.2, m003 → all 0.3, m004 → all 0.4
    """
    return pd.DataFrame({
        "MaterialId":           ["m001", "m002", "m003", "m004"],
        "Composition":          ["FeO", "Fe2O3", "TiO2", "NiO"],
        "Reduced Formula":      ["FeO", "Fe2O3", "TiO2", "NiO"],
        "Elements":             [["Fe", "O"], ["Fe", "O"], ["Ti", "O"], ["Ni", "O"]],
        "NSites":               [2, 5, 3, 2],
        "Bandgap":              [0.0, 1.2, 3.0, 0.0],
        "Dimensionality Cheon": ["3D", "3D", "2D", "3D"],
        "Disorder Probability": [0.1, 0.5, 0.05, 0.9],
        "gga_only_pbx_save_id": ["GGA_m001", "GGA_m002", "GGA_m003", "GGA_m004"],
        "mixed_pbx_save_id":    ["GGA_m001", "Not computed", "GGA_m003", "GGA_m004"],
    })


@pytest.fixture
def h5_path_with_data(tmp_path):
    """HDF5 file with 4 entries matching synthetic_df pbx save IDs.

    "S100" is an HDF5 fixed-length byte-string type (100 chars max) — the same
    encoding used by the real screening databases.  AQ_H5Database.read_id()
    decodes these bytes back to Python str automatically.

    Entry i has all cells set to 0.1*(i+1):
        GGA_m001 → 0.1, GGA_m002 → 0.2, GGA_m003 → 0.3, GGA_m004 → 0.4
    """
    filepath = tmp_path / "test.h5"
    ids = ["GGA_m001", "GGA_m002", "GGA_m003", "GGA_m004"]
    data = np.stack([np.full(ARRAY_SHAPE, 0.1 * (i + 1)) for i in range(len(ids))])
    with h5py.File(filepath, "w") as f:
        f.create_dataset("ids", data=np.array(ids, dtype="S100"))
        f.create_dataset("decomposition_energies", data=data)
    return filepath
