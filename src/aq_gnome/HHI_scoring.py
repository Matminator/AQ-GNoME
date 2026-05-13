import numpy as np
import pandas as pd
from pathlib import Path
from pymatgen.core import Composition


class Compound_HHI_scores:

    def __init__(self, path_to_data_folder=None):
        if path_to_data_folder is None:
            package_root = Path(__file__).resolve().parent
            root = package_root.parent.parent  # go up 2 levels
            data_path = root / "data"
        self.HHI_df = pd.read_csv(data_path / 'HHI_values.csv')

    def _compute_HHI_score(self, element_dict: dict[str, int | float], HHI_type: str, exclude_elements: list[str]):
        keys = list(element_dict.keys())
        average_HHI = 0
        N_atoms = 0
        for key in keys:
            if key not in exclude_elements:
                masked_df = self.HHI_df[self.HHI_df['Element'] == key]
                if len(masked_df) != 0:
                    average_HHI += masked_df['HHI_' + HHI_type].values[0] * element_dict[key]
                    N_atoms += element_dict[key]

        if N_atoms == 0:
            return 0
        return int(np.round(average_HHI / N_atoms, 0))

    def get_max_HHI_score(self, row, HHI_type: str = 'P', exclude_elements: list[str] = []):
        comp = Composition(row['Composition'])
        element_dict = comp.get_el_amt_dict()

        keys = list(element_dict.keys())
        max_HHI = 0
        for key in keys:
            masked_df = self.HHI_df[self.HHI_df['Element'] == key]
            if len(masked_df) != 0:
                HHI_value = masked_df['HHI_' + HHI_type].values[0]
                if HHI_value > max_HHI:
                    max_HHI = HHI_value

        return int(max_HHI)

    def get_HHI_score(self, row, HHI_type: str = 'P', exclude_elements: list[str] = []):
        comp = Composition(row['Composition'])
        element_dict = comp.get_el_amt_dict()
        return self._compute_HHI_score(element_dict, HHI_type, exclude_elements)

    def get_HHI_dummys(self, dummy_formula, HHI_type: str = 'P', exclude_elements: list[str] = []):
        if dummy_formula == 'IrO2':
            element_dict = {'Ir': 1, 'O': 2}
        elif dummy_formula == 'TiO2':
            element_dict = {'Ti': 1, 'O': 2}
        elif dummy_formula == 'RuO2':
            element_dict = {'Ru': 1, 'O': 2}
        elif dummy_formula == 'ZrO2':
            element_dict = {'Zr': 1, 'O': 2}
        else:
            raise ValueError("Dummy formula not recognized:", dummy_formula)

        return self._compute_HHI_score(element_dict, HHI_type, exclude_elements)
