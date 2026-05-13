# AQ-GNoME

AQ-GNoME evaluates the aqueous electrochemical (Pourbaix) stability of crystal structures from the GNoME database. The package provides tools for screening, filtering, and ranking candidates by stability criteria, elemental composition, Herfindahl–Hirschman Index (HHI) scores, and crystallographic disorder probability.

## Installation

```bash
git clone https://github.com/Matminator/AQ-GNoME.git
cd AQ-GNoME
pip install -e .
```

## Data

The required data files can be downloaded from:
https://doi.org/10.11583/DTU.30738716

## Usage

See `data_analasis_v2.ipynb` for a worked example.

## Citing this work

If you use AQ-GNoME, please also cite the following:

**GNoME** (crystal structure data):
Merchant et al., *Nature* (2023). https://doi.org/10.1038/s41586-023-06735-9
GitHub: https://github.com/google-deepmind/materials_discovery

**Disorder predictions** (if using the Disorder Probability column):
Jakob et al., *Advanced Materials* (2025). https://doi.org/10.1002/adma.202514226
"Learning Crystallographic Disorder: Bridging Prediction and Experiment in Materials Discovery"

**HHI scores** (if using the HHI columns):
Gaultois et al., *Chem. Mater.* 25, 2911–2920 (2013). https://doi.org/10.1021/cm400893e

## License

MIT License — Copyright © 2026 Technical University of Denmark. See [LICENSE](LICENSE).
