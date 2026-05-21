# AQ-GNoME

AQ-GNoME evaluates the aqueous electrochemical (Pourbaix) stability of crystal structures from the GNoME database. The package provides tools for screening, filtering, and ranking candidates by stability criteria, elemental composition, Herfindahl–Hirschman Index (HHI) scores, and crystallographic disorder probability.

## Installation

```bash
git clone https://github.com/Matminator/AQ-GNoME.git
cd AQ-GNoME
pip install -e .
```

## Data

Download `data.zip` from the DTU Data deposit and unzip it into the root of the repository:

https://doi.org/10.11583/DTU.30738716

The `data/` folder is not included in this repository.

## Usage

See [`notebooks/data_analysis_v2.ipynb`](notebooks/data_analysis_v2.ipynb) for a worked example of the full screening pipeline.

## Citing this work

If you use AQ-GNoME, please cite:

Nissen, M. S., Beck, P., Karlsson, L., et al. AQ-GNoME: an Aqueous Stability Augmentation of the GNoME Database. *ChemRxiv* (2026). https://doi.org/10.26434/chemrxiv.15003677/v1

Please also cite the following:

**GNoME** (crystal structure data):
Merchant et al., *Nature* (2023). https://doi.org/10.1038/s41586-023-06735-9
GitHub: https://github.com/google-deepmind/materials_discovery

**Disorder predictions** (if using the Disorder Probability column):
Jakob et al., *Advanced Materials* (2025). https://doi.org/10.1002/adma.202514226
"Learning Crystallographic Disorder: Bridging Prediction and Experiment in Materials Discovery"

**HHI scores** (if using the HHI columns):
Gaultois et al., *Chem. Mater.* 25, 2911–2920 (2013). https://doi.org/10.1021/cm400893e

## License

**Code:** MIT License — Copyright © 2026 Technical University of Denmark. See [LICENSE](LICENSE).

**Data:** The dataset (available at https://doi.org/10.11583/DTU.30738716) is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) due to the inclusion of GNoME-derived data (Google DeepMind, CC BY-NC 4.0). Non-commercial use only. See the [DTU Data deposit](https://doi.org/10.11583/DTU.30738716) for per-file license and attribution details.
