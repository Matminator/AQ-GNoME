import pandas as pd
from pathlib import Path
import ast

from aq_gnome.aq_stability_h5py_database_setup import AQ_H5Database

CSV_GGAONLY_SOLID_FILTER_TRUE = 'stable_materials_summary_combinedGGA_only_2_filter_True.csv'
CSV_GGAONLY_SOLID_FILTER_FALSE = 'stable_materials_summary_combinedGGA_only_2_filter_False.csv'
CSV_MIXED_SOLID_FILTER_TRUE = 'stable_materials_summary_combinedMixed_2_filter_True.csv'
CSV_MIXED_SOLID_FILTER_FALSE = 'stable_materials_summary_combinedMixed_2_filter_False.csv'

H5PY_DB_GGAONLY_SOLID_FILTER_TRUE = 'screening_results_GGA_only_2_filter_True.h5'
H5PY_DB_GGAONLY_SOLID_FILTER_FALSE = 'screening_results_GGA_only_2_filter_False.h5'
H5PY_DB_MIXED_SOLID_FILTER_TRUE = 'screening_results_Mixed_2_filter_True.h5'
H5PY_DB_MIXED_SOLID_FILTER_FALSE = 'screening_results_Mixed_2_filter_False.h5'

class Data_Handler:

    def __str__(self):
        return f"Data_Handler(solid_filter={self.solid_filter}, gga_only={self.gga_only})"

    def __init__(self, solid_filter: bool, gga_only: bool, 
                path_to_data_directory: str | Path | None = None):
        self.solid_filter = solid_filter
        self.gga_only = gga_only

        if path_to_data_directory is None:
            package_root = Path(__file__).resolve().parent
            root = package_root.parent.parent  # go up 2 levels
            data_path = root / "data"
        else:
            if isinstance(path_to_data_directory, str):
                data_path = Path(path_to_data_directory)
            if not data_path.is_dir():
                raise NotADirectoryError(f"The provided path_to_data_directory is not a directory: {data_path}")
            
            data_path = data_path.resolve()

        # Load CSVs, which are mostly that found from the GNoME github:
        gga_df = pd.read_csv(
            data_path /
            (CSV_GGAONLY_SOLID_FILTER_TRUE if solid_filter else CSV_GGAONLY_SOLID_FILTER_FALSE),
            index_col=0
            )
        mixed_df = pd.read_csv(
            data_path /
            (CSV_MIXED_SOLID_FILTER_TRUE if solid_filter else CSV_MIXED_SOLID_FILTER_FALSE),
            index_col=0
            )
        disorder = pd.read_csv(data_path / "disorder_prob.csv", index_col=0)
        precomputed_HHI = pd.read_csv(data_path / "precomputed_HHI_scores.csv", index_col=0)

        gga_df = gga_df.rename(columns={"pbx_save_id": "gga_only_pbx_save_id"})
        mixed_df = mixed_df.rename(columns={"pbx_save_id": "mixed_pbx_save_id"})
        combined_df = gga_df.merge(mixed_df[["MaterialId", "mixed_pbx_save_id"]], on="MaterialId")
        combined_df = combined_df.merge(disorder[["MaterialId", "Disorder Probability"]], on="MaterialId")
        combined_df = combined_df.merge(precomputed_HHI, on="MaterialId")

        self.N_total_GNoME = len(combined_df)
        combined_df["Elements"] = combined_df["Elements"].apply(ast.literal_eval)
        self.combined_df = combined_df[combined_df['gga_only_pbx_save_id'] != 'Not computed']
        self.modified_df = self.combined_df.copy()

        # Load H5PY databases, which contain the computed decomposition
        # energies:
        self.gga_results = AQ_H5Database(
            data_path / 
            (H5PY_DB_GGAONLY_SOLID_FILTER_TRUE if solid_filter else H5PY_DB_GGAONLY_SOLID_FILTER_FALSE),
            mode='r')
        self.mixed_results = AQ_H5Database(
            data_path /
            (H5PY_DB_MIXED_SOLID_FILTER_TRUE if solid_filter else H5PY_DB_MIXED_SOLID_FILTER_FALSE),
            mode='r')
        
    def remove_entries_with_elements(self, elements: str | list[str]):
        if isinstance(elements, str):
            elements = [elements]
            
        df = self.modified_df.copy()
        df = df[~df['Elements'].apply(lambda x: any(i in elements for i in x))]
        print("Number of entries removed:", len(self.modified_df) - len(df),
              'Number of entries left:', len(df), 'which is',
              f"{len(df) / self.N_total_GNoME * 100:.2f}% of total GNoME database" )
        self.modified_df = df

    def remove_entries_not_consisting_exclusively_of_elements(self, elements: str | list[str]):
        if isinstance(elements, str):
            elements = [elements]

        df = self.modified_df.copy()
        df = df[df['Elements'].apply(lambda x: all(i in elements for i in x))]
        print(
            'Removed all entries not consisting exclusively of:', ', '.join(elements),
            '. Number of entries removed:', len(self.modified_df) - len(df),
            'Number of entries left:', len(df), 'which is',
            f"{len(df) / self.N_total_GNoME * 100:.2f}% of total GNoME database"
        )
        self.modified_df = df

    def remove_entries_without_elements(self, elements: str | list[str],
                                        must_contain_all: bool):
        if isinstance(elements, str):
            elements = [elements]

        df = self.modified_df.copy()
            
        if must_contain_all:
            for el in elements:
                if len(df[df['Elements'].apply(lambda x: any(i == el for i in x))]) == 0:
                    df = df.head(0)
                    message = 'Removed all entries not containing all of the following elements: ' + ', '.join(elements)
                    break
                df = df[df['Elements'].apply(lambda x: any(i == el for i in x))]
                message = 'Removed all entries not containing all of the following elements: ' + ', '.join(elements)
        else:
            if len(df[df['Elements'].apply(lambda x: any(i in elements for i in x))]) == 0:
                df = df.head(0)
                message = 'Removed all entries not containing any of the following elements: ' + ', '.join(elements)
            df = df[df['Elements'].apply(lambda x: any(i in elements for i in x))]
            message = 'Removed all entries not containing any of the following elements: ' + ', '.join(elements)
            
        print(
            message,
            ".Number of entries removed:", len(self.modified_df) - len(df),
            'Number of entries left:', len(df), 'which is',
            f"{len(df) / self.N_total_GNoME * 100:.2f}% of total GNoME database"
            )
        
        self.modified_df = df

    def restore_df(self):
        self.modified_df = self.combined_df.copy()
    
    def get_df(self):
        return self.modified_df.copy()
