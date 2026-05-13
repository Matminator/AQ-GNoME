from mp_api.client import MPRester


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
