import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from aq_gnome.data_handler import Data_Handler


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
            structures_db=None):

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
                    round(self.max_dG_per_id[mid][sc_idx], 2)
                    for mid in self.stable_df['MaterialId']
                ])
        return self.stable_df.copy()

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
