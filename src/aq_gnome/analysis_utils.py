import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
import ase.db
from pymatgen.core import Composition
from pathlib import Path
from aq_gnome.aq_stability_h5py_database_setup import AQ_H5Database
from aq_gnome.data_utils import Data_Handler
from ase.db import connect


def get_col_dict_for_atoms(elements_to_discart=[], elements_to_include=[], elements_exclusively_allowed=[]):
    elements_cols = {}
    for element in elements_exclusively_allowed:
        elements_cols[element] = 'tab:blue'
    for element in elements_to_include:
        elements_cols[element] = 'tab:green'
    for element in elements_to_discart:
        elements_cols[element] = 'tab:red'
    return elements_cols

def get_col_dict_for_each_list_of_elements(list_of_elements: list[list[str]]):
    import seaborn as sns
    col_list = sns.color_palette("tab10", 100)
    elements_cols = {}
    for i, elements in enumerate(list_of_elements):
        color = col_list[i]
        for element in elements:
            elements_cols[element] = color
    return elements_cols            

def plot_periodic_table_with_values(elements_cols: dict):

    # Periodic table layout: element symbol -> (group, period)
    positions = {
        # Period 1
        'H': (1, 1), 'He': (18, 1),
        # Period 2
        'Li': (1, 2), 'Be': (2, 2),
        'B': (13, 2), 'C': (14, 2), 'N': (15, 2), 'O': (16, 2), 'F': (17, 2), 'Ne': (18, 2),
        # Period 3
        'Na': (1, 3), 'Mg': (2, 3),
        'Al': (13, 3), 'Si': (14, 3), 'P': (15, 3), 'S': (16, 3), 'Cl': (17, 3), 'Ar': (18, 3),
        # Period 4
        'K': (1, 4), 'Ca': (2, 4), 'Sc': (3, 4), 'Ti': (4, 4), 'V': (5, 4), 'Cr': (6, 4), 'Mn': (7, 4), 'Fe': (8, 4),
        'Co': (9, 4), 'Ni': (10, 4), 'Cu': (11, 4), 'Zn': (12, 4),
        'Ga': (13, 4), 'Ge': (14, 4), 'As': (15, 4), 'Se': (16, 4), 'Br': (17, 4), 'Kr': (18, 4),
        # Period 5
        'Rb': (1, 5), 'Sr': (2, 5), 'Y': (3, 5), 'Zr': (4, 5), 'Nb': (5, 5), 'Mo': (6, 5), 'Tc': (7, 5), 'Ru': (8, 5),
        'Rh': (9, 5), 'Pd': (10, 5), 'Ag': (11, 5), 'Cd': (12, 5),
        'In': (13, 5), 'Sn': (14, 5), 'Sb': (15, 5), 'Te': (16, 5), 'I': (17, 5), 'Xe': (18, 5),
        # Period 6
        'Cs': (1, 6), 'Ba': (2, 6), 'La': (3, 9), 'Ce': (4, 9), 'Pr': (5, 9), 'Nd': (6, 9), 'Pm': (7, 9), 'Sm': (8, 9),
        'Eu': (9, 9), 'Gd': (10, 9), 'Tb': (11, 9), 'Dy': (12, 9), 'Ho': (13, 9), 'Er': (14, 9),
        'Tm': (15, 9), 'Yb': (16, 9), 'Lu': (17, 9),
        'Hf': (4, 6), 'Ta': (5, 6), 'W': (6, 6), 'Re': (7, 6), 'Os': (8, 6), 'Ir': (9, 6), 'Pt': (10, 6),
        'Au': (11, 6), 'Hg': (12, 6), 'Tl': (13, 6), 'Pb': (14, 6), 'Bi': (15, 6), 'Po': (16, 6), 'At': (17, 6), 'Rn': (18, 6),
        # Period 7
        'Fr': (1, 7), 'Ra': (2, 7), 'Ac': (3, 10), 'Th': (4, 10), 'Pa': (5, 10), 'U': (6, 10), 'Np': (7, 10), 'Pu': (8, 10),
        'Am': (9, 10), 'Cm': (10, 10), 'Bk': (11, 10), 'Cf': (12, 10), 'Es': (13, 10), 'Fm': (14, 10), 'Md': (15, 10),
        'No': (16, 10), 'Lr': (17, 10),
        'Rf': (4, 7), 'Db': (5, 7), 'Sg': (6, 7), 'Bh': (7, 7), 'Hs': (8, 7), 'Mt': (9, 7), 'Ds': (10, 7),
        'Rg': (11, 7), 'Cn': (12, 7), 'Nh': (13, 7), 'Fl': (14, 7), 'Mc': (15, 7), 'Lv': (16, 7), 'Ts': (17, 7), 'Og': (18, 7)
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0.5, 18.5)
    ax.set_ylim(11, 0.5)
    ax.axis('off')

    # Draw each element cell
    for symbol, (x, y) in positions.items():
        if symbol in elements_cols:
            color = elements_cols[symbol]
        else:
            color =  (0.9, 0.9, 0.9, 1.0)  # gray if no value
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, symbol, ha='center', va='center', fontsize=14, fontweight='bold')

    # Add colorbar
    plt.tight_layout()
    plt.show()

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

class atoms_from_db():

    def __init__(self, db_path = None):
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
    
    # --- new methods --------------------------------------------------

    def get_atoms_material_id(self, material_id: str, 
                                  df: pd.DataFrame):
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


class Stability_Criteria:

    def __init__(self, Us: int | float | list[int] | list[float], 
                       pHs: int | float | list[int] | list[float],
                       decomposition_threshold: float = 0.5):
        
        for p in [Us, pHs]:
            if isinstance(p, list):
                if len(p) == 1:
                    p = p[0]
                elif len(p) > 2:
                    raise ValueError("Us/pHs list can have at most 2 elements. Gave:", p)
                elif len(p) < 1:
                    raise ValueError("Us/pHs list must have at least 1 element. Gave:", p)

        self.Us = Us
        self.pHs = pHs
        self.decomposition_threshold = decomposition_threshold
        self.U_interval = np.linspace(-2, 4, 31) # Data specific 
        self.pH_interval = np.linspace(-2, 16, 19) # Data specific

        if isinstance(Us, list):
            self.U_lowlimt_index = self._closest_index(self.U_interval, min(Us)+1e-6, method='floor')
            self.U_highlimt_index = self._closest_index(self.U_interval, max(Us)-1e-6, method='ceil') + 1
            self.U_index = None
        else:
            self.U_lowlimt_index = None
            self.U_highlimt_index = None
            self.U_index = self._closest_index(self.U_interval, Us, method='closest')
        if isinstance(pHs, list):
            self.pH_lowlimt_index = self._closest_index(self.pH_interval, min(pHs)+1e-6, method='floor')
            self.pH_highlimt_index = self._closest_index(self.pH_interval, max(pHs)-1e-6, method='ceil') + 1
            self.pH_index = None
        else:
            self.pH_lowlimt_index = None
            self.pH_highlimt_index = None
            self.pH_index = self._closest_index(self.pH_interval, pHs, method='closest')

        # Column name used when appending max_dG_in_region results to a DataFrame
        Us_str = (str(self.Us[0]) if min(self.Us) == max(self.Us)
                  else f"[{min(self.Us)},{max(self.Us)}]") if isinstance(self.Us, list) else str(self.Us)
        pHs_str = (str(self.pHs[0]) if min(self.pHs) == max(self.pHs)
                   else f"[{min(self.pHs)},{max(self.pHs)}]") if isinstance(self.pHs, list) else str(self.pHs)
        self.col_name = f"max_dG_U{Us_str}_pH{pHs_str}"

    def _closest_index(self, arr: NDArray[np.float64], x: float, method: str = 'closest'):
        if method == 'closest':
            result = np.abs(arr - x).argmin()
        elif method == 'floor':
            result = np.max(np.where(arr <= x)[0].max(initial=-1))
        elif method == 'ceil':
            result = np.min(np.where(arr >= x)[0])
        return result

    def max_dG_in_region(self, decom_G: NDArray[np.float64]) -> float:
        """Return the max decomposition energy in the U×pH region this criterion covers."""
        if isinstance(self.Us, list) and isinstance(self.pHs, list):
            region = decom_G[self.U_lowlimt_index:self.U_highlimt_index,
                             self.pH_lowlimt_index:self.pH_highlimt_index]
        elif isinstance(self.Us, list):
            region = decom_G[self.U_lowlimt_index:self.U_highlimt_index, self.pH_index]
        elif isinstance(self.pHs, list):
            region = decom_G[self.U_index, self.pH_lowlimt_index:self.pH_highlimt_index]
        else:
            region = decom_G[self.U_index, self.pH_index]
        return float(np.max(region))


    def visualize(self):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        matrix = np.ones((len(self.U_interval), len(self.pH_interval)))
        if isinstance(self.Us, list) and isinstance(self.pHs, list):
            matrix[self.U_lowlimt_index:self.U_highlimt_index,
                   self.pH_lowlimt_index:self.pH_highlimt_index] = 0
        elif isinstance(self.Us, list):
            matrix[self.U_lowlimt_index:self.U_highlimt_index, self.pH_index] = 0
        elif isinstance(self.pHs, list):
            matrix[self.U_index, self.pH_lowlimt_index:self.pH_highlimt_index] = 0
        else:
            matrix[self.U_index, self.pH_index] = 0

        ax.imshow(matrix, origin='lower', vmin=0, cmap='terrain') 
        ax.set_title(r'$\Delta$G$_{decomp}$ <= ' + str(self.decomposition_threshold) + ' eV/atom')
        ax.set_xticks(np.linspace(0,len(self.pH_interval)-1, 10), [-2, 0, 2, 4, 6, 8, 10, 12, 14, 16])
        ax.set_xlabel('pH')
        ax.set_yticks(np.linspace(0,len(self.U_interval)-1, 7), [-2, -1, 0, 1, 2, 3, 4])
        ax.set_ylabel('Potential [V]')
        ax.set_aspect(0.5)
        

class Stable_Entries:
    def __init__(self,
            data_handler: Data_Handler,
            stability_criteria: Stability_Criteria | list[Stability_Criteria],
            structures_db = None):

        if type(stability_criteria) is not list:
            stability_criteria = [stability_criteria]
        self.stability_criteria = stability_criteria
        self.structures_db = structures_db

        self.df = data_handler.get_df()
        self.results_gga = data_handler.gga_results
        if data_handler.gga_only:
            self.results_mixed = None
            print("Only GGA/GGA(+U) results will be used for stability analysis.")
        else:
            self.results_mixed = data_handler.mixed_results
            print("Mixed GGA/GGA(+U)/r2SCAN results will be used for stability analysis.")

        self.stable_df = None
        self.ids_of_stable_entries = None
        self.max_dG_per_id = None  # {material_id: [max_dG_sc0, max_dG_sc1, ...]} for stable entries

    def get_stable_df(self):
        if self.stable_df is None:
            if not self.ids_of_stable_entries:
                self.find_stable_entries()
            stable_ids = self.ids_of_stable_entries
            self.stable_df = self.df[self.df['MaterialId'].isin(stable_ids)].copy()
            for sc_idx, sc in enumerate(self.stability_criteria):
                self.stable_df.insert(sc_idx, sc.col_name, [
                    self.max_dG_per_id[mid][sc_idx]
                    for mid in self.stable_df['MaterialId']
                ])
        return self.stable_df.copy()
    
    def get_stable_atoms(self):
        if self.structures_db is None:
            raise ValueError("ASE database must be provided to get stable Atoms objects.")
        
        df = self.get_stable_df()
        out_atoms = []
        for i, row in df.iterrows():
            raise NotImplementedError("This have not been updated yet")
            entry = self.structures_db.read_id(row['pbx_save_id'])
            atoms = AseAtomsAdaptor.get_atoms(entry.structure)
            out_atoms.append(atoms)
            
        return out_atoms
    
    def get_decom_G(self, row):

        if self.results_mixed:
            mixed_pbx_id = row['mixed_pbx_save_id']
            if mixed_pbx_id != 'Not computed':
                decom_G = self.results_mixed.read_id(row['mixed_pbx_save_id'])
        if not self.results_mixed or mixed_pbx_id == 'Not computed':
            decom_G = self.results_gga.read_id(row['gga_only_pbx_save_id'])
        return decom_G

    def find_stable_entries(self):
        if self.ids_of_stable_entries is not None:
            return self.ids_of_stable_entries.copy()
        
        ids_of_stable_entries = []
        max_dG_per_id = {}
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            failed_stability = False

            decom_G = self.get_decom_G(row)

            for stability_criterion in self.stability_criteria:
                if stability_criterion.max_dG_in_region(decom_G) > stability_criterion.decomposition_threshold:
                    failed_stability = True
                    break

            if not failed_stability:
                mid = row['MaterialId']
                ids_of_stable_entries.append(mid)
                max_dG_per_id[mid] = [sc.max_dG_in_region(decom_G) for sc in self.stability_criteria]

        self.ids_of_stable_entries = ids_of_stable_entries
        self.max_dG_per_id = max_dG_per_id
        print(f"Found {len(self.ids_of_stable_entries)} stable entries given the stability criteria.")


class sys_in_MP_db:
    def __init__(self, api_key: str):
        self.mpr = MPRester(api_key)

    def in_db(self, sys):
        sys_exists = False
        input_set = set(sys)
        gga_entries = self.mpr.get_entries_in_chemsys(sys,
                additional_criteria={"thermo_types": ["GGA_GGA+U","GGA_GGA+U_R2SCAN","R2SCAN"]})
        for entry in gga_entries:
            set_of_elements = set(e.symbol for e in entry.elements)
            if set_of_elements == input_set:

                sys_exists = True
                break
        return sys_exists
    

class Compound_HHI_scores:

    def __init__(self, path_to_data_folder = None):
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


        
