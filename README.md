# ERA5 Extreme Rainfall Nowcasting - Aceh

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine learning-based nowcasting of 3-hour extreme rainfall over Aceh using ERA5 single-level reanalysis.  
**Task formulation**: Binary classification on a 0.25° grid around Banda Aceh.

---

## Project Overview

This project builds a reproducible pipeline to:
- Load and prepare ERA5 reanalysis data (2020–2024)
- Engineer time-series and temporal features
- Define extreme events using the 95th percentile of 3-hour ahead precipitation
- Train baseline and ensemble models
- Evaluate with chronological split to emulate an operational setting

---

## Study Area and Data

**Region**  
- Latitude: 5.0°N to 6.0°N  
- Longitude: 95.0°E to 96.0°E  
- Grid resolution: 0.25° (producing 5×5 grid cells)

**Temporal Coverage**  
- Resolution: 3-hourly  
- Period: 2020–2024  
- Dataset size: 365,400 samples

**Data Source**  
Copernicus Climate Data Store - ERA5 Single Levels Reanalysis  
[https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)

**Variables Used**  
| Variable | Description |
|----------|-------------|
| `tp` | Total precipitation |
| `ro` | Runoff |
| `u10`, `v10` | 10m wind components |
| `t2m` | 2m temperature |
| `swvl1` | Top layer soil moisture |

---

## Labels

**Target**: Extreme rainfall 3 hours ahead

- `tp_next` is the 3-hour ahead accumulated precipitation
- **Extreme event** is defined as `tp_next ≥ empirical 95th percentile`
- Label name: `is_extreme_next`

---

## Features

Implemented features include:

**Current Meteorology**  
- `tp`, `ro`, `u10`, `v10`, `t2m`, `swvl1`  
- Derived: `wind_speed`

**Precipitation History**  
- `tp_lag1`, `tp_lag2`, `tp_roll3_mean`

**Temporal Encoding**  
- `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`

**Spatial Coordinates**  
- `latitude`, `longitude`

---

## Models

- **Logistic Regression** as baseline
- **Random Forest** as main model

**Evaluation Strategy**: Chronological split by year
- **Train**: 2020–2022
- **Validation**: 2023
- **Test**: 2024

---

## Repository Structure

```
era5-extreme-rainfall-aceh/
├── README.md
├── requirements.txt
├── .gitignore
├── reports/
│   ├── proposal_report_extreme_rainfall_aceh.pdf
│   ├── progress_report_extreme_rainfall_aceh.pdf
│   └── final_report_extreme_rainfall_aceh.pdf
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   └── model_training.py
├── data/
│   └── README.md
├── images/
│   └── figures/
└── slides/
    └── presentation.pdf
```

**Note**: Large raw data should not be committed. Use links or release assets if needed.

---

## How to Run

### Requirements
- Python 3.10 or newer
- pandas, numpy, scikit-learn, matplotlib

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/era5-extreme-rainfall-aceh.git
cd era5-extreme-rainfall-aceh

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Notebooks
1. Open `notebooks/` and run in order
2. Ensure paths point to the processed dataset CSV

### Optional: Run Streamlit Demo

```bash
streamlit run app.py
```

---

## Results Summary

Current progress indicates that non-linear ensembles improve discrimination of rare extreme events compared with linear baselines. Detailed experiments and metrics are provided in the final report.

---

## Reports

All reports follow IEEE format:

| Report | Pages | Content |
|--------|-------|---------|
| Proposal | 2 (excl. references) | Problem statement, methodology |
| Progress | 3 (excl. references) | Preliminary experiments, next steps |
| Final | Up to 10 | Full experiments, discussion, future work |

---

## Team

- Muhammad Raza Adzani
- Ahmad Siddiq

---

## License and Attribution

ERA5 data are provided by ECMWF via the Copernicus Climate Data Store.  
Please cite the original papers listed in the reports when reusing the methodology.

This project is licensed under the MIT License.

---

## Contact

**Email**: raza.a22@mhs.usk.ac.id

---

## Acknowledgments

- ECMWF and Copernicus Climate Data Store for ERA5 data
- Universitas Syiah Kuala for academic support
