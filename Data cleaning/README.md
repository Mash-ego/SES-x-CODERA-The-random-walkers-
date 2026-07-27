# Multi-Predictor Economic Indicator Analysis

This folder is a self-contained, portable copy of `multi_predictor_analysis.py`.
Follow the steps below on the new computer to get it running.

## 1. Prerequisites

- **Python 3.9 or newer** installed on the target machine.
  - Check with: `python3 --version` (Mac/Linux) or `python --version` (Windows)
  - If not installed, get it from https://www.python.org/downloads/

You do NOT need to install pandas/numpy/matplotlib/statsmodels manually —
the setup script below does that for you, inside an isolated environment
(a "virtual environment") that won't interfere with anything else on the
computer.

## 2. One-time setup

**Mac / Linux:**
```bash
cd multi_predictor_package
bash setup.sh
```

**Windows:**
Double-click `setup.bat`, or run from a terminal:
```
cd multi_predictor_package
setup.bat
```

This creates a `.venv` folder containing Python plus the exact libraries
needed (pandas, numpy, matplotlib, statsmodels — versions pinned in
`requirements.txt`).

## 3. Add the data files

The script does NOT include the underlying data — only the analysis code.
Place the following CSV files directly in this same folder (next to
`multi_predictor_analysis.py`), before running it:

| File name (must match exactly) | Source |
|---|---|
| `ECONDATA_MARKET_RATES_USDZAR.csv` | EconData/SARB export |
| `CPIAUCNS.csv` | FRED export |
| `FEDFUNDS.csv` | FRED export |
| `ECONDATA_CPI_ANL_SERIES(2.2.1).csv` | EconData export |
| `ECONDATA_MARKET_RATES_REPO.csv` | EconData export |
| `VIXCLS.csv` | FRED export |
| `Platinum_Futures_Historical_Data - Platinum_Futures_Historical_Data.csv` | Investing.com-style export |

If a filename or set of indicators differs, edit the `INDICATORS` dict near
the top of `multi_predictor_analysis.py` to point at whatever files you have.

## 4. Run it

**Mac / Linux:**
```bash
source .venv/bin/activate
python multi_predictor_analysis.py
```

**Windows:**
```
.venv\Scripts\activate.bat
python multi_predictor_analysis.py
```

Output plots and CSVs will appear in a new `indicator_plots/` folder created
next to the script.

## 5. Re-running later

You only need to run `setup.sh` / `setup.bat` once. After that, just activate
the environment (the `source .venv/bin/activate` or `.venv\Scripts\activate.bat`
line above) each time you open a new terminal, then run the script.

## Folder contents

```
multi_predictor_package/
├── multi_predictor_analysis.py   <- the analysis script
├── requirements.txt              <- exact library versions needed
├── setup.sh                      <- one-time setup for Mac/Linux
├── setup.bat                     <- one-time setup for Windows
├── README.md                     <- this file
└── (place your CSV data files here)
```
