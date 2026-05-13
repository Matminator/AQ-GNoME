import h5py
import numpy as np

class AQ_H5Database:

    def __init__(self, filename: str, mode: str):
        self.closed = False
        self.filename = filename
        self.mode = mode

        self.h5file = h5py.File(filename, mode)
        if mode == 'r':
            ids = self.h5file["ids"][:]
            ids_as_str = [id_str.decode("utf-8") 
                                 if isinstance(id_str, bytes) 
                                 else id_str
                                 for id_str in ids]
            self.ids_to_index = {id_str: idx for idx, id_str 
                                 in enumerate(ids_as_str)}

    def close(self):
        self.h5file.close()
        self.closed = True

    def add_item(self, id_str, array):
        if self.closed or self.mode != 'a':
            raise ValueError("Database is closed or not in append " \
            "mode. Make new instance.")

        dset_arrays = self.h5file["decomposition_energies"]
        dset_ids = self.h5file["ids"]
        
        n = dset_arrays.shape[0]
        
        # Resize both datasets
        dset_arrays.resize(n + 1, axis=0)
        dset_ids.resize(n + 1, axis=0)
        
        # Store
        dset_arrays[n] = array
        dset_ids[n] = id_str

    def read_id(self, id):
        if self.closed or self.mode != 'r':
            raise ValueError("Database is closed or not in read mode." \
            " Make new instance.")

        idx = self.ids_to_index[id]
        return self.h5file["decomposition_energies"][idx]