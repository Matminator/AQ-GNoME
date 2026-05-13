import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ase.db import connect


def get_simplified_df(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    for name in ['gga_only_pbx_save_id', 'mixed_pbx_save_id', 'Data Directory',
                'Is Train', 'Decomposition Energy Per Atom Relative',
                'Decomposition Energy Per Atom All',
                'Decomposition Energy Per Atom MP', 'Decomposition Energy Per Atom MP OQMD',
                'Decomposition Energy Per Atom',
                'Corrected Energy', 'Space Group Number',
                'Uncorrected Energy', 'Space Group',
                'Density', 'Volume', 'Point Group', 'Formation Energy Per Atom'
                ]:
        try:
            df_copy = df_copy.drop(columns=[name])
        except KeyError:
            pass

    return df_copy


class atoms_from_db:

    def __init__(self, db_path=None):
        if db_path is None:
            package_root = Path(__file__).resolve().parent
            root = package_root.parent.parent  # go up 2 levels
            db_path = root / "data/ase_db_GNoME_pp_corr.db"
        if isinstance(db_path, str):
            db_path = Path(db_path)
        if db_path.is_dir():
            db_path = db_path.resolve() / "ase_db_GNoME_pp_corr.db"

        self.db = connect(db_path)

    def get_atoms_objects_from_df(self, df: pd.DataFrame):
        out_atoms = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            db_row = self.db.get(id_key=row['gga_only_pbx_save_id'])
            atoms = db_row.toatoms()
            out_atoms.append(atoms)
        return out_atoms

    def get_atoms_material_id(self, material_id: str, df: pd.DataFrame):
        row = df[df['MaterialId'] == material_id].iloc[0]
        db_row = self.db.get(id_key=row['gga_only_pbx_save_id'])
        atoms = db_row.toatoms()
        return atoms

    def get_atoms_from_row(self, row: pd.Series):
        db_row = self.db.get(id_key=row['gga_only_pbx_save_id'])
        atoms = db_row.toatoms()
        return atoms

    def get_atoms_from_pbx_save_id(self, pbx_save_id):
        db_row = self.db.get(id_key=pbx_save_id)
        atoms = db_row.toatoms()
        return atoms
