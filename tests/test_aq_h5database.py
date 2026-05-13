"""
Tests for AQ_H5Database (aq_stability_h5py_database_setup.py)
==============================================================

Background
----------
AQ_H5Database is a thin wrapper around an HDF5 file that stores precomputed
Pourbaix decomposition energies for every GNoME material.  The file contains
two datasets:

  ids                    — 1-D array of string IDs (one per material)
  decomposition_energies — 2-D array per material, shape (31, 19), matching
                           the U × pH grid used by Stability_Criteria

The class is opened in read mode ('r') for normal use and in append mode
('a') when new results are being written.

These tests use pytest's `tmp_path` fixture to create a minimal HDF5 file
in a temporary directory — no part of the real data/ folder is needed.
"""

import numpy as np
import pytest
import h5py

from aq_gnome.aq_stability_h5py_database_setup import AQ_H5Database

ARRAY_SHAPE = (31, 19)  # U-grid × pH-grid, same as Stability_Criteria
IDS = ["id_001", "id_002", "id_003"]


# ── Fixture ─────────────────────────────────────────────────────────────────

@pytest.fixture
def h5_path(tmp_path):
    """
    Create a minimal readable HDF5 file in a temporary directory.

    The file contains:
      - ids                    : the three string IDs in IDS
      - decomposition_energies : three (31, 19) arrays filled with known values
                                 (0.1, 0.2, 0.3) so tests can verify the
                                 correct row is returned.
    """
    filepath = tmp_path / "test_db.h5"
    data = np.stack([np.full(ARRAY_SHAPE, 0.1 * (i + 1)) for i in range(len(IDS))])

    with h5py.File(filepath, "w") as f:
        f.create_dataset("ids", data=np.array(IDS, dtype="S100"))
        f.create_dataset("decomposition_energies", data=data)

    return filepath


# ── Tests ────────────────────────────────────────────────────────────────────

def test_read_id_returns_array(h5_path):
    """
    read_id() should return the numpy array stored for the given ID.

    The fixture stores id_001 → all-0.1, id_002 → all-0.2, id_003 → all-0.3.
    This test verifies:
      - the return type is a numpy array
      - the shape is (31, 19)  (the U × pH grid)
      - the values match what was written (confirming the correct row is read)
    """
    db = AQ_H5Database(h5_path, mode="r")
    result = db.read_id("id_002")

    assert isinstance(result, np.ndarray)
    assert result.shape == ARRAY_SHAPE
    assert np.allclose(result, 0.2)

    db.close()


def test_read_unknown_id_raises(h5_path):
    """
    Requesting an ID that does not exist in the database should raise KeyError.
    The lookup is done via a dict built at open time, so any missing key raises
    KeyError before touching the HDF5 file.
    """
    db = AQ_H5Database(h5_path, mode="r")

    with pytest.raises(KeyError):
        db.read_id("nonexistent_id")

    db.close()


def test_closed_db_raises(h5_path):
    """
    After close() is called, the database should refuse further reads and raise
    ValueError.  This guards against accidental use of a stale handle.
    """
    db = AQ_H5Database(h5_path, mode="r")
    db.close()

    with pytest.raises(ValueError):
        db.read_id("id_001")


def test_wrong_mode_add_item_raises(h5_path):
    """
    add_item() requires the database to be opened in append mode ('a').
    Calling it on a read-mode ('r') database should raise ValueError
    immediately, without touching the file contents.
    """
    db = AQ_H5Database(h5_path, mode="r")

    with pytest.raises(ValueError):
        db.add_item("new_id", np.zeros(ARRAY_SHAPE))

    db.close()
