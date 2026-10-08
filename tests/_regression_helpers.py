import ast
import warnings
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SKIP_COL = "Data Directory"


def load_snapshot(name: str) -> pd.DataFrame:
    """Load a CSV snapshot; issue a warning and skip the test if the file is missing."""
    path = SNAPSHOT_DIR / f"{name}.csv"
    if not path.exists():
        warnings.warn(
            f"Snapshot '{name}.csv' not found in tests/snapshots/. "
            "Re-run the save-snapshot cells in the notebooks to create it.",
            stacklevel=2,
        )
        pytest.skip(f"Snapshot '{name}.csv' missing — see warning above")
    df = pd.read_csv(path)
    if "Elements" in df.columns:
        df["Elements"] = df["Elements"].apply(ast.literal_eval)
    return df


def assert_pipeline_result(actual: pd.DataFrame, snapshot_name: str, label: str) -> None:
    """Compare actual DataFrame against a saved CSV snapshot.

    ALL columns are compared:
    - numeric cols: tolerance rtol=1e-3 (safe for CSV float precision)
    - string cols (Composition, Reduced Formula, Crystal System, …): exact match
    - Elements (list of str): exact match, order-sensitive

    Error reporting:
    - MaterialId mismatches → lists missing / extra materials with their IDs
    - Numeric column failures → lists each differing column with max deviation
    - Non-numeric column failures → re-raises the pandas AssertionError unchanged
    """
    expected = load_snapshot(snapshot_name)

    def _norm(df):
        return (df.drop(columns=[SKIP_COL], errors="ignore")
                  .sort_values("MaterialId")
                  .reset_index(drop=True))

    actual_norm = _norm(actual)
    expected_norm = _norm(expected)

    # Step 1 — check MaterialId sets before comparing values
    actual_ids = set(actual_norm["MaterialId"])
    expected_ids = set(expected_norm["MaterialId"])
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        lines = [
            f"{label}: candidate set mismatch "
            f"({len(actual_ids)} actual, {len(expected_ids)} expected)"
        ]
        if missing:
            lines.append(
                f"  Missing ({len(missing)}): {missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )
        if extra:
            lines.append(
                f"  Extra   ({len(extra)}): {extra[:10]}"
                + (" ..." if len(extra) > 10 else "")
            )
        pytest.fail("\n".join(lines))

    # Step 2 — full row comparison across all columns
    try:
        pd.testing.assert_frame_equal(
            actual_norm,
            expected_norm,
            check_like=True,
            rtol=1e-3,
            check_exact=False,
            obj=label,
        )
    except AssertionError as err:
        # Pinpoint which numeric columns differ (extra context on top of pandas' message)
        bad_cols = []
        for col in actual_norm.select_dtypes(include=[np.number]).columns:
            if col not in expected_norm.columns:
                continue
            a = actual_norm[col].to_numpy(float, na_value=np.nan)
            e = expected_norm[col].to_numpy(float, na_value=np.nan)
            if not np.allclose(a, e, rtol=1e-3, equal_nan=True):
                bad_cols.append(
                    f"    {col}: max deviation = {np.nanmax(np.abs(a - e)):.3e}"
                )
        if bad_cols:
            pytest.fail(
                f"{label}: numeric values differ in {len(bad_cols)} column(s):\n"
                + "\n".join(bad_cols)
                + f"\n\n{err}"
            )
        # Non-numeric failure (e.g. Reduced Formula or Elements mismatch):
        # re-raise unchanged — the pandas message is already informative.
        raise
