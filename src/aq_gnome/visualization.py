import matplotlib.pyplot as plt


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

    for symbol, (x, y) in positions.items():
        if symbol in elements_cols:
            color = elements_cols[symbol]
        else:
            color = (0.9, 0.9, 0.9, 1.0)
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, symbol, ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
