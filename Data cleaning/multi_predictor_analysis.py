"""
Exploratory Data Analysis — Multiple Economic Indicators (weekly, Friday freq)
================================================================================

WHAT THIS SCRIPT DOES
----------------------
Runs the SAME exploratory pipeline that was originally built for USDZAR on
ANY number of economic indicators — e.g. USDZAR, US CPI, the Fed Funds
rate, or any other EconData/FRED series. Just add each series to the
INDICATORS dict below, pointing at its raw CSV export, and the script will
produce the full set of plots for every one of them.

SUPPORTED INPUT FORMATS (auto-detected per file)
---------------------------------------------------
1. EconData exports (e.g. SARB USDZAR EXCX135) — a metadata block at the
   top followed by headerless "date,value" rows.
2. FRED (Federal Reserve Economic Data) exports (e.g. CPIAUCNS, FEDFUNDS)
   — a clean two-column CSV with a proper header row:
        observation_date,CPIAUCNS
        1913-01-01,9.800

For each indicator, it:
1. Loads the raw CSV, auto-detecting which of the two formats above it is.
2. Filters to your chosen date range (year/month/day precision).
3. Resamples to WEEKLY, FRIDAY frequency. Series published less often than
   weekly (like CPI, which is monthly) are forward-filled — i.e. each
   Friday uses the most recently PUBLISHED value as of that date, which is
   the correct "real-time" way to line up low-frequency data with the
   Friday forecasting cadence (no lookahead).
4. Prints/visualizes the resulting dataframe (head, tail, summary stats).
5. Produces the full set of exploratory plots:
      - Level series over time
      - % change (returns)
      - Distribution/histogram of % change
      - Zig-zag indicator
      - ACF test (autocorrelation function graph + Ljung-Box significance test)
      - Scatterplot with best fit line (pairing USDZAR with a predictor)
   All output files are prefixed with the indicator name and saved to
   OUTPUT_DIR, e.g. "USDZAR_level.png", "CPI_US_level.png".

HOW TO ADD A NEW INDICATOR
----------------------------
1. Export the series as CSV — either from EconData, or from FRED
2. Add an entry to the INDICATORS dict below:

    "CPI_SA": {
        "path": "ECONDATA_CPI.csv",
        "label": "SA CPI Index",
    },

REQUIREMENTS
-------------
pip install pandas numpy matplotlib statsmodels --break-system-packages
"""
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Directory this script lives in. Used to resolve the CSV paths in
# INDICATORS below so the script works no matter what folder you launch
# it FROM (e.g. running it via an IDE "Run" button, a scheduled task, or
# `python some/other/dir/multi_predictor_analysis.py` from elsewhere) —
# it always looks for the CSVs sitting next to the script itself.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
# Add / remove predictors here. Each needs a raw EconData/FED CSV export.
# Paths can be just the filename (resolved relative to this script's folder,
# see SCRIPT_DIR above) or a full absolute path if your CSVs live elsewhere.
INDICATORS = {
    "USDZAR": {
        "path": "ECONDATA_MARKET_RATES_USDZAR.csv",
        "label": "Rand per USD",
    },
    "CPI_US": {
        "path": "CPIAUCNS.csv",
        "label": "US CPI (Index, NSA)",
    },
    "FED_Rate": {
        "path": "FEDFUNDS.csv",
        "label": "Effective Federal Funds Rate (%)",
    },
     "CPI_SA": {
         "path": "ECONDATA_CPI_ANL_SERIES(2.2.1).csv",
         "label": "SA CPI Index",
     },
     "REPO_SA": {
         "path": "ECONDATA_MARKET_RATES_REPO.csv",
         "label": "SA Repo Rate (%)",
     },
     "VIX": {
              "path": "VIXCLS.csv",
              "label": "VIX (%)",
          },
    "Platinum_Futures_Price": {
             "path": "Platinum_Futures_Historical_Data - Platinum_Futures_Historical_Data.csv",
             "label": "Platinum Futures Price ($)",
         },
}

START_YEAR = 2000                 # first year of data to keep (inclusive)
START_MONTH = 1
START_DAY = 1

END_YEAR = 2025                    # last year of data to keep (inclusive)
END_MONTH = 12
END_DAY = 31

ZIGZAG_THRESHOLD = 0.05            # 5% minimum reversal to register a new pivot

# Indicators for which the ADF "differenced" test and the ACF test should be
# run on LOG RETURNS (log(value).diff()) rather than a plain first difference
# of the level. This is standard for price-like series (an exchange rate, a
# commodity price, a volatility index) where the % change is the economically
# meaningful, roughly stationary quantity — as opposed to something like an
# interest rate or an index level, where a plain level difference is more
# natural. Add/remove keys here as needed; anything not listed keeps using
# the first-difference-of-level test.
LOG_RETURN_SERIES = ["USDZAR", "VIX", "Platinum_Futures_Price"]

# Pairs of indicators to scatter-plot against each other (x, y).
# Each pair must reference keys from INDICATORS above. Leave empty to skip.
SCATTER_PAIRS = [
    ("CPI_US", "USDZAR"),
    ("CPI_SA", "USDZAR"),
]

# Pairs of indicators to compute a DIFFERENTIAL for (indicator1 - indicator2),
# e.g. a US-SA inflation differential. Each pair must reference keys from
# INDICATORS above. Leave empty to skip.
DIFFERENTIAL_PAIRS = [
    ("CPI_US", "CPI_SA"),("FED_Rate", "REPO_SA")
]

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "indicator_plots")  # all plots/csvs are saved in here


# ----------------------------------------------------------------------
# 1. LOAD DATA (generic — works for daily, weekly, or monthly EconData series)
# ----------------------------------------------------------------------
def load_indicator(path: str) -> pd.DataFrame:
    """
    Loads a time series CSV from EITHER of two source formats and returns a
    dataframe with a single generic column "value", indexed by date:

    FORMAT 1 — EconData exports (e.g. SARB USDZAR):
        Has a metadata block at the top (series name, frequency, unit, etc.
        as "Key,Value" rows) followed by plain "date,value" rows with NO
        header of their own, e.g.:

            ECONDATA:...,SOME_CODE
            Label,...
            ...
            Unit of measure,...
            2010-01-04,7.4130
            2010-01-05,7.3800

    FORMAT 2 — FRED (Federal Reserve Economic Data) exports (e.g. CPI,
    Fed Funds Rate): a clean two-column CSV with a proper header row:

            observation_date,CPIAUCNS
            1913-01-01,9.800
            1913-02-01,9.800

    The function auto-detects which format it's looking at by checking
    whether the first row is a valid "date,value" pair (FRED) or plain
    text metadata (EconData), and parses accordingly. Frequency is NOT
    forced here — whatever native frequency the series was published at
    (daily, monthly, etc.) is preserved; resample_weekly_friday handles
    converting it to weekly-Friday later.
    """
    # If a bare/relative filename was given, resolve it relative to this
    # script's own folder rather than whatever the current working
    # directory happens to be (fixes FileNotFoundError when the script is
    # run from a different directory, e.g. via an IDE or shortcut).
    if not os.path.isabs(path):
        candidate = os.path.join(SCRIPT_DIR, path)
        if os.path.exists(candidate):
            path = candidate

    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # Check if the very first row already looks like "date,value" (FRED format)
    first_fields = lines[0].strip().split(",")
    first_cell_is_date = pd.notna(
        pd.to_datetime(first_fields[0].strip(), errors="coerce", format="%Y-%m-%d")
    )

    if first_cell_is_date:
        # Shouldn't normally happen (FRED has a header row), but handle it
        # just in case a headerless file is passed in directly.
        df = pd.read_csv(path, header=None, names=["date", "value"], encoding="utf-8-sig")
    else:
        # Try reading it as a standard CSV with a header row first (FRED format:
        # "observation_date,<SERIES_CODE>")
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            raw = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            looks_like_fred = (
                raw is not None and raw.shape[1] == 2 and
                pd.to_datetime(raw.iloc[:, 0], errors="coerce").notna().mean() > 0.9
            )

        if looks_like_fred:
            # FRED-style: first column is dates, second is the value, real header exists
            df = raw.copy()
            df.columns = ["date", "value"]
        else:
            # Fall back to EconData-style: scan for the first row where the
            # first field parses as a date, and treat everything from there
            # as headerless date,value data.
            data_start = None
            for i, line in enumerate(lines):
                first_field = line.split(",")[0].strip()
                parsed = pd.to_datetime(first_field, errors="coerce", format="%Y-%m-%d")
                if pd.notna(parsed):
                    data_start = i
                    break

            if data_start is None:
                raise ValueError(
                    f"Could not parse {path} as either EconData or FRED format. "
                    "Check the file structure."
                )

            df = pd.read_csv(
                path,
                skiprows=data_start,
                header=None,
                names=["date", "value"],
                encoding="utf-8-sig",
            )

    # NOTE: use flexible parsing here (no forced format). The FRED-detection
    # step above also parses flexibly, so a file that passed that check (e.g.
    # one using MM/DD/YYYY-style dates instead of ISO YYYY-MM-DD) must be
    # parsed the same flexible way here too — otherwise every date silently
    # becomes NaT and dropna() below wipes out the entire file.
    n_before = len(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    if df.empty:
        raise ValueError(
            f"{path}: parsed 0 valid rows out of {n_before} raw rows. "
            "This almost always means the date or value format wasn't "
            "recognized (check for an unexpected date format, thousands "
            "separators in the value column, etc.)."
        )
    return df


def filter_by_date_range(df: pd.DataFrame,
                          start_year: int, start_month: int = 1, start_day: int = 1,
                          end_year: int = None, end_month: int = 12, end_day: int = 31) -> pd.DataFrame:
    """
    Restricts the dataframe to observations between a chosen start date
    and end date (inclusive), specified as separate year/month/day values.
    Clips to the available data range (with a warning) if the request goes
    beyond what's on file, and rolls invalid days (e.g. Feb 30) to the
    nearest valid date.
    """
    if end_year is None:
        end_year = start_year

    try:
        start_date = pd.Timestamp(year=start_year, month=start_month, day=start_day)
    except ValueError:
        start_date = pd.Timestamp(year=start_year, month=start_month, day=1) + pd.offsets.MonthEnd(0)
        print(f"Warning: invalid start day, using {start_date.date()} instead.")

    try:
        end_date = pd.Timestamp(year=end_year, month=end_month, day=end_day)
    except ValueError:
        end_date = pd.Timestamp(year=end_year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
        print(f"Warning: invalid end day, using {end_date.date()} instead.")

    if start_date > end_date:
        raise ValueError(f"start_date {start_date.date()} is after end_date {end_date.date()}.")

    if start_date < df.index.min():
        print(f"Warning: requested start {start_date.date()} is before the "
              f"earliest available data ({df.index.min().date()}). "
              f"Using {df.index.min().date()} instead.")
        start_date = df.index.min()

    if end_date > df.index.max():
        print(f"Warning: requested end {end_date.date()} is after the "
              f"latest available data ({df.index.max().date()}). "
              f"Using {df.index.max().date()} instead.")
        end_date = df.index.max()

    filtered = df.loc[start_date:end_date]
    print(f"Filtered to {start_date.date()} - {end_date.date()}: "
          f"{len(filtered)} observations")
    return filtered


# ----------------------------------------------------------------------
# 2. RESAMPLE TO WEEKLY-FRIDAY FREQUENCY (works for any native frequency)
# ----------------------------------------------------------------------
def resample_weekly_friday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses ANY frequency series (daily, weekly, monthly) to one
    observation per week, anchored on Friday (W-FRI).

    - For daily/high-frequency series: takes the last value observed in
      each week (e.g. the Friday close).
    - For low-frequency series (e.g. monthly CPI): most weeks won't have a
      new observation, so the value is forward-filled — each Friday shows
      the most recently PUBLISHED value as of that date. This is the
      correct real-time treatment: it never looks ahead to a CPI print
      that hadn't been released yet as of a given Friday.
    """
    weekly = df["value"].resample("W-FRI").last()
    weekly = weekly.ffill()   # carries forward low-frequency series (e.g. monthly CPI)
    weekly = weekly.to_frame(name="value")
    weekly = weekly.dropna()
    weekly["log_return"] = np.log(weekly["value"]).diff()
    weekly["pct_change"] = weekly["value"].pct_change() * 100
    return weekly


# ----------------------------------------------------------------------
# 3. VISUALIZE THE DATAFRAME ITSELF
# ----------------------------------------------------------------------
def show_dataframe_summary(weekly: pd.DataFrame, name: str, label: str, outdir: str):
    print("=" * 60)
    print(f"{name} — WEEKLY (FRIDAY) DATAFRAME — HEAD")
    print("=" * 60)
    print(weekly.head(10))

    print(f"\n{name} — TAIL")
    print(weekly.tail(10))

    print(f"\n{name} — SUMMARY STATISTICS")
    print(weekly.describe())

    print(f"\nTotal weekly observations: {len(weekly)}")
    print(f"Date range: {weekly.index.min().date()} to {weekly.index.max().date()}")
    print(f"Missing values:\n{weekly.isna().sum()}")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=np.round(weekly.head(8).values, 4),
        colLabels=weekly.columns,
        rowLabels=[d.strftime("%Y-%m-%d") for d in weekly.head(8).index],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax.set_title(f"{name} — weekly (Friday), first 8 rows ({label})", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_dataframe_preview.png"), dpi=150, bbox_inches="tight")
    plt.close()

def run_adf_test(series: pd.Series, name: str, alpha: float = 0.05) -> dict:
    """
    Runs the Augmented Dickey-Fuller (ADF) test for stationarity on a
    time series (e.g. the weekly USDZAR level, or its returns).

    The ADF test's null hypothesis (H0) is that the series HAS a unit
    root, i.e. it is NON-stationary (has a trend / random-walk behaviour).
    The alternative (H1) is that the series IS stationary.

    - If p-value <= alpha: reject H0 -> series is likely STATIONARY.
    - If p-value >  alpha: fail to reject H0 -> series is likely
      NON-STATIONARY (which is the typical, expected result for a raw
      exchange rate level like USDZAR — it usually only becomes
      stationary after differencing, e.g. using returns instead of levels).

    Prints a plain-English summary and returns the key results as a dict
    so they can be reused elsewhere (e.g. included in a report table).

    Example:
        run_adf_test(weekly["value"], "USDZAR (level)")
        run_adf_test(weekly["value"].diff().dropna(), "USDZAR (first difference)")
    """
    series = series.dropna()
    if len(series) < 10:
        print(f"{name}: not enough observations to run the ADF test. Skipping.")
        return {}

    result = adfuller(series, autolag="AIC")
    adf_stat, p_value, used_lag, n_obs, crit_values, icbest = result

    is_stationary = p_value <= alpha

    print(f"\nADF Test — {name}")
    print("-" * 50)
    print(f"ADF statistic:     {adf_stat:.4f}")
    print(f"p-value:           {p_value:.4f}")
    print(f"Lags used:         {used_lag}")
    print(f"Observations used: {n_obs}")
    print("Critical values:")
    for key, val in crit_values.items():
        print(f"    {key}: {val:.4f}")

    verdict = "STATIONARY" if is_stationary else "NON-STATIONARY"
    print(f"Verdict (alpha={alpha}): p-value {'<=' if is_stationary else '>'} {alpha} "
          f"-> series is likely {verdict}")

    return {
        "name": name,
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "lags_used": used_lag,
        "n_obs": n_obs,
        "critical_values": crit_values,
        "is_stationary": is_stationary,
    }


def run_acf_test(series: pd.Series, name: str, outdir: str,
                  lags: int = 20, alpha: float = 0.05) -> pd.DataFrame:
    """
    Runs an ACF (autocorrelation function) test on a time series and
    produces a dedicated ACF graph.

    "Test" here means two things, both standard for checking
    autocorrelation:
    1. Computes the ACF value at each lag, along with its (1-alpha)
       confidence interval (via statsmodels' acf(..., alpha=alpha)) — a
       lag is "significant" if its ACF value falls outside the confidence
       band, i.e. it's unlikely to be zero by chance.
    2. Runs the Ljung-Box test (via acorr_ljungbox) at each lag, whose
       null hypothesis (H0) is that the data are independently
       distributed (no autocorrelation up to that lag). A small p-value
       (<= alpha) means we reject H0 -> there IS significant
       autocorrelation up to that lag.

    Saves a bar-style ACF plot with the confidence band shaded
    (as "{name}_acf_test.png") and a CSV of the per-lag results
    (as "{name}_acf_test_results.csv"), and returns the results as a
    dataframe with columns: lag, acf, ci_lower, ci_upper, significant,
    ljung_box_stat, ljung_box_pvalue.

    Example:
        run_acf_test(weekly["log_return"], "USDZAR", outdir, lags=20)
    """
    series = series.dropna()
    if len(series) <= lags:
        print(f"{name}: not enough observations for an ACF test at {lags} lags. Skipping.")
        return pd.DataFrame()

    acf_values, conf_int = acf(series, nlags=lags, alpha=alpha, fft=False)
    ci_lower = conf_int[:, 0] - acf_values
    ci_upper = conf_int[:, 1] - acf_values

    lb_result = acorr_ljungbox(series, lags=range(1, lags + 1), return_df=True)

    results = pd.DataFrame({
        "lag": range(0, lags + 1),
        "acf": acf_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    })
    results["significant"] = (results["acf"] < ci_lower) | (results["acf"] > ci_upper)
    results.loc[results["lag"] == 0, "significant"] = False  # lag 0 (ACF=1) is trivial, not a real test

    # Ljung-Box results start at lag 1 (lag 0 is trivially 1.0 and not tested)
    results.loc[results["lag"] >= 1, "ljung_box_stat"] = lb_result["lb_stat"].values
    results.loc[results["lag"] >= 1, "ljung_box_pvalue"] = lb_result["lb_pvalue"].values

    # --- plot ---
    plt.figure(figsize=(10, 5))
    x = results["lag"].values
    plt.bar(x, results["acf"].values, width=0.3, color="#2a6fdb", zorder=3)
    plt.fill_between(x, ci_lower, ci_upper, color="#2a6fdb", alpha=0.15,
                      label=f"{int((1-alpha)*100)}% confidence band")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"{name} — Autocorrelation Function (ACF), {lags} lags")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.legend()
    plt.grid(alpha=0.3, zorder=0)
    plt.tight_layout()
    plot_path = os.path.join(outdir, f"{name}_acf_test.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    csv_path = os.path.join(outdir, f"{name}_acf_test_results.csv")
    results.to_csv(csv_path, index=False)

    n_sig = int(results.loc[results["lag"] >= 1, "significant"].sum())
    print(f"\nACF Test — {name}")
    print("-" * 50)
    print(f"Lags tested: {lags}")
    print(f"Significant lags (outside {int((1-alpha)*100)}% band): {n_sig} of {lags}")
    if not lb_result.empty:
        first_lb_p = lb_result['lb_pvalue'].iloc[0]
        overall_verdict = "significant autocorrelation" if first_lb_p <= alpha else "no significant autocorrelation"
        print(f"Ljung-Box test at lag 1: p-value = {first_lb_p:.4f} -> {overall_verdict}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {csv_path}")

    return results


# ----------------------------------------------------------------------
# 4. EXPLORATORY PLOTS (all generic — work for any predictor)
# ----------------------------------------------------------------------
def plot_level_series(weekly: pd.DataFrame, name: str, label: str, outdir: str):
    plt.figure(figsize=(10, 5))
    plt.plot(weekly.index, weekly["value"], color="#2a6fdb")
    plt.title(f"{name} — Weekly (Friday) Level")
    plt.xlabel("Date")
    plt.ylabel(label)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_level.png"), dpi=150)
    plt.close()


def plot_returns(weekly: pd.DataFrame, name: str, outdir: str):
    plt.figure(figsize=(10, 4))
    plt.plot(weekly.index, weekly["pct_change"], color="#d62728", linewidth=0.9)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"{name} — Weekly % Change")
    plt.xlabel("Date")
    plt.ylabel("% change")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_returns.png"), dpi=150)
    plt.close()


def plot_return_distribution(weekly: pd.DataFrame, name: str, outdir: str):
    plt.figure(figsize=(7, 5))
    plt.hist(weekly["pct_change"].dropna(), bins=40, color="#2a6fdb", edgecolor="white")
    plt.title(f"{name} — Distribution of Weekly % Changes")
    plt.xlabel("% change")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_returns_distribution.png"), dpi=150)
    plt.close()

def compute_zigzag(series: pd.Series, threshold: float = 0.05) -> pd.Series:
    """
    Computes zig-zag pivot points for any price/level series. Filters out
    small "noise" moves and only marks a new pivot once the series has
    reversed by more than `threshold` (as a fraction, e.g. 0.05 = 5%) from
    the last pivot. Returns a Series (same index) that is NaN everywhere
    except at the identified pivot points.
    """
    series = series.dropna()
    dates = series.index
    values = series.values

    pivot_dates = [dates[0]]
    pivot_values = [values[0]]
    trend = None
    last_pivot_value = values[0]

    for i in range(1, len(values)):
        val = values[i]
        change = (val - last_pivot_value) / last_pivot_value

        if trend is None:
            if abs(change) >= threshold:
                trend = "up" if change > 0 else "down"
                pivot_dates.append(dates[i])
                pivot_values.append(val)
                last_pivot_value = val

        elif trend == "up":
            if val >= last_pivot_value:
                pivot_dates[-1] = dates[i]
                pivot_values[-1] = val
                last_pivot_value = val
            elif change <= -threshold:
                trend = "down"
                pivot_dates.append(dates[i])
                pivot_values.append(val)
                last_pivot_value = val

        elif trend == "down":
            if val <= last_pivot_value:
                pivot_dates[-1] = dates[i]
                pivot_values[-1] = val
                last_pivot_value = val
            elif change >= threshold:
                trend = "up"
                pivot_dates.append(dates[i])
                pivot_values.append(val)
                last_pivot_value = val

    zigzag = pd.Series(index=dates, dtype=float)
    zigzag.loc[pivot_dates] = pivot_values
    return zigzag


def plot_zigzag(weekly: pd.DataFrame, name: str, label: str, outdir: str, threshold: float = 0.05):
    """
    Plots the raw series with the zig-zag indicator overlaid, connecting
    only the confirmed swing highs/lows.
    """
    zz = compute_zigzag(weekly["value"], threshold=threshold)
    zz_points = zz.dropna()
    if len(zz_points) < 2:
        print(f"{name}: threshold {threshold:.0%} too high for this series "
              f"(only {len(zz_points)} pivot found). Skipping zig-zag plot.")
        return

    plt.figure(figsize=(11, 5.5))
    plt.plot(weekly.index, weekly["value"], color="#bbbbbb",
              linewidth=1, label=f"{name} (weekly)")
    plt.plot(zz_points.index, zz_points.values, color="#d62728",
              linewidth=1.6, marker="o", markersize=4,
              label=f"Zig-zag ({threshold:.0%} threshold)")

    plt.title(f"{name} Zig-Zag Indicator (reversal threshold = {threshold:.0%})")
    plt.xlabel("Date")
    plt.ylabel(label)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_zigzag_indicator.png"), dpi=150)
    plt.close()

    print(f"{name}: zig-zag identified {len(zz_points)} pivot points "
          f"(threshold = {threshold:.0%})")


def plot_scatter_with_fit(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str,
                           title: str, outpath: str):
    """
    Generic scatterplot of y vs x with an OLS best-fit line overlaid.
    Both x and y should be pandas Series (any two aligned numeric series —
    e.g. one indicator vs another, or a series vs its own lag).

    Aligns x and y on their shared index automatically, drops any rows
    with missing values in either, fits a simple linear regression
    (y = slope*x + intercept) via numpy.polyfit, and annotates the plot
    with the fitted equation and R².
    """
    aligned = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(aligned) < 3:
        print(f"Not enough overlapping observations to fit a line for '{title}'. Skipping.")
        return

    xs = aligned["x"].values
    ys = aligned["y"].values

    slope, intercept = np.polyfit(xs, ys, 1)
    y_pred = slope * xs + intercept
    ss_res = np.sum((ys - y_pred) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    x_line = np.linspace(xs.min(), xs.max(), 100)
    y_line = slope * x_line + intercept

    plt.figure(figsize=(7.5, 6))
    plt.scatter(xs, ys, s=14, alpha=0.5, color="#2a6fdb", label="Observations")
    plt.plot(x_line, y_line, color="#d62728", linewidth=2,
              label=f"Best fit: y = {slope:.4f}x + {intercept:.2f}\nR\u00b2 = {r_squared:.3f}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    print(f"Saved scatter: {outpath}  "
          f"(n={len(aligned)}, slope={slope:.4f}, R\u00b2={r_squared:.3f})")


def plot_lag_scatter(weekly: pd.DataFrame, name: str, outdir: str, lag: int = 1):
    """
    Scatterplot of the series against its own value `lag` weeks earlier,
    with a best-fit line — a quick visual check of how strongly (and
    linearly) the series is related to its recent past. This is the same
    relationship an AR(lag) model is trying to capture.

    Example: plot_lag_scatter(weekly, "USDZAR", outdir, lag=1) plots
    this week's USDZAR against last week's USDZAR.
    """
    series = weekly["value"]
    lagged = series.shift(lag)
    outpath = os.path.join(outdir, f"{name}_lag{lag}_scatter.png")
    plot_scatter_with_fit(
        x=lagged, y=series,
        xlabel=f"{name}, t-{lag}",
        ylabel=f"{name}, t",
        title=f"{name} — Value vs {lag}-Week Lag (with best fit line)",
        outpath=outpath,
    )


def plot_cross_indicator_scatter(all_weekly: dict, x_name: str, y_name: str, outdir: str):
    """
    Scatterplot of one indicator against another (e.g. CPI vs USDZAR),
    aligned on their shared weekly-Friday dates, with a best-fit line.
    Useful for a quick visual gut-check of whether two series co-move
    before building a model that uses one to help forecast the other.

    Example: plot_cross_indicator_scatter(all_weekly, "CPI_US", "USDZAR", outdir)
    """
    if x_name not in all_weekly or y_name not in all_weekly:
        print(f"Skipping scatter {x_name} vs {y_name}: one or both indicators "
              f"not found in processed data.")
        return

    x_series = all_weekly[x_name]["value"]
    y_series = all_weekly[y_name]["value"]
    outpath = os.path.join(outdir, f"{x_name}_vs_{y_name}_scatter.png")
    plot_scatter_with_fit(
        x=x_series, y=y_series,
        xlabel=x_name,
        ylabel=y_name,
        title=f"{y_name} vs {x_name} (weekly, Friday) \u2014 with best fit line",
        outpath=outpath,
    )

def compute_indicator_differential(all_weekly: dict, name1: str, name2: str, outdir: str) -> pd.DataFrame:
    """
    Computes the differential between two indicators (name1 - name2),
    aligned on their shared weekly-Friday dates. Produces two versions of
    the differential, since both are useful for different purposes:

    1. LEVEL differential: value1 - value2
       e.g. CPI_US - CPI_SA (in index points — not usually meaningful on
       its own since CPI indices use different base years, but included
       for completeness).

    2. INFLATION-STYLE differential: pct_change(value1) - pct_change(value2)
       e.g. US weekly CPI % change minus SA weekly CPI % change — this is
       the economically meaningful one. A US-SA inflation differential is
       a classic Purchasing Power Parity (PPP) style predictor for
       USDZAR: if SA inflation runs persistently higher than US inflation,
       basic PPP theory says the Rand should tend to depreciate over time
       to compensate.

    Saves a plot of both differentials over time, plus a CSV of the
    underlying numbers, and returns the differential dataframe so it can
    be reused (e.g. as a candidate-model regressor).

    Example:
        diff = compute_indicator_differential(all_weekly, "CPI_US", "CPI_SA", outdir)
    """
    if name1 not in all_weekly or name2 not in all_weekly:
        print(f"Skipping differential {name1} - {name2}: one or both indicators "
              f"not found in processed data. (Add the missing series to "
              f"INDICATORS and re-run.)")
        return pd.DataFrame()

    df1 = all_weekly[name1]
    df2 = all_weekly[name2]

    aligned = pd.concat([
        df1["value"].rename(f"{name1}_value"),
        df2["value"].rename(f"{name2}_value"),
        df1["pct_change"].rename(f"{name1}_pct_change"),
        df2["pct_change"].rename(f"{name2}_pct_change"),
    ], axis=1).dropna()

    if aligned.empty:
        print(f"Skipping differential {name1} - {name2}: no overlapping dates "
              f"between the two series.")
        return pd.DataFrame()

    aligned["level_differential"] = aligned[f"{name1}_value"] - aligned[f"{name2}_value"]
    aligned["pct_change_differential"] = (
        aligned[f"{name1}_pct_change"] - aligned[f"{name2}_pct_change"]
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(aligned.index, aligned["level_differential"], color="#2a6fdb")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"{name1} \u2212 {name2} \u2014 Level Differential")
    axes[0].grid(alpha=0.3)

    axes[1].plot(aligned.index, aligned["pct_change_differential"], color="#d62728")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title(f"{name1} \u2212 {name2} \u2014 Weekly % Change Differential "
                       f"(\u201cinflation differential\u201d style)")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(outdir, f"{name1}_minus_{name2}_differential.png")
    plt.savefig(outpath, dpi=150)
    plt.close()

    csv_path = os.path.join(outdir, f"{name1}_minus_{name2}_differential.csv")
    aligned[["level_differential", "pct_change_differential"]].to_csv(csv_path)

    print(f"Saved differential: {outpath}")
    print(f"  Mean level differential:      {aligned['level_differential'].mean():.4f}")
    print(f"  Mean pct-change differential: {aligned['pct_change_differential'].mean():.4f}")

    return aligned

# ----------------------------------------------------------------------
# 5. RUN THE FULL PIPELINE FOR ONE INDICATOR
# ----------------------------------------------------------------------
def run_pipeline_for_indicator(name: str, path: str, label: str, outdir: str):
    print("\n" + "#" * 70)
    print(f"# {name}")
    print("#" * 70)

    df = load_indicator(path)
    df = filter_by_date_range(df, START_YEAR, START_MONTH, START_DAY,
                               END_YEAR, END_MONTH, END_DAY)
    weekly = resample_weekly_friday(df)

    if weekly.empty:
        raise ValueError(
            f"{name}: weekly dataframe is empty after resampling — nothing "
            f"to plot. Check that '{path}' actually contains data inside the "
            f"configured date range ({START_YEAR}-{START_MONTH:02d}-{START_DAY:02d} "
            f"to {END_YEAR}-{END_MONTH:02d}-{END_DAY:02d})."
        )

    show_dataframe_summary(weekly, name, label, outdir)

    use_log_returns = name in LOG_RETURN_SERIES
    adf_level = run_adf_test(weekly["value"], f"{name} (level)")
    if use_log_returns:
        # Price-like series (USDZAR, VIX, Platinum, ...): test stationarity
        # of the log return rather than the raw level difference.
        adf_diff = run_adf_test(weekly["log_return"].dropna(), f"{name} (log return)")
    else:
        adf_diff = run_adf_test(weekly["value"].diff().dropna(), f"{name} (first difference)")
    adf_results = pd.DataFrame([r for r in [adf_level, adf_diff] if r])
    if not adf_results.empty:
        adf_results.to_csv(os.path.join(outdir, f"{name}_adf_test_results.csv"), index=False)

    # ACF/Ljung-Box test: run on log returns (already computed for every
    # series in resample_weekly_friday()) — this is the standard, comparable
    # transform for price-like series such as USDZAR, VIX, and Platinum.
    acf_results = run_acf_test(weekly["log_return"], name, outdir, lags=20)

    plot_level_series(weekly, name, label, outdir)
    plot_returns(weekly, name, outdir)
    plot_return_distribution(weekly, name, outdir)
    plot_zigzag(weekly, name, label, outdir, threshold=ZIGZAG_THRESHOLD)
    plot_lag_scatter(weekly, name, outdir, lag=1)

    weekly.to_csv(os.path.join(outdir, f"{name}_weekly_friday.csv"))
    return {
        "weekly": weekly,
        "adf_level": adf_level,
        "adf_diff": adf_diff,
        "acf_results": acf_results,
    }


def build_test_summary(all_results: dict, outdir: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Builds ONE consolidated summary table across ALL indicators, pulling
    together every ADF and ACF test result computed in the pipeline. This
    is meant to be the single reference table for a report/presentation —
    rather than having to flip between each indicator's individual test
    output, everything is side by side, one row per indicator.

    Columns:
        indicator
        adf_level_stat, adf_level_pvalue, adf_level_stationary
        adf_diff_stat,  adf_diff_pvalue,  adf_diff_stationary
        acf_n_significant_lags   -- out of the lags tested (excl. lag 0)
        acf_lags_tested
        acf_ljungbox_lag1_pvalue
        acf_has_autocorrelation  -- True if Ljung-Box p-value at lag 1 <= alpha

    Saves the table to "test_summary.csv" and a companion bar chart
    ("test_summary_pvalues.png") comparing the key p-values against the
    alpha threshold for every indicator, and returns the table as a
    dataframe.

    Example:
        all_results = {}
        for name, cfg in INDICATORS.items():
            all_results[name] = run_pipeline_for_indicator(name, cfg["path"], cfg["label"], OUTPUT_DIR)
        summary = build_test_summary(all_results, OUTPUT_DIR)
    """
    rows = []
    for name, res in all_results.items():
        adf_level = res.get("adf_level") or {}
        adf_diff = res.get("adf_diff") or {}
        acf_df = res.get("acf_results")

        row = {
            "indicator": name,
            "adf_level_stat": adf_level.get("adf_statistic"),
            "adf_level_pvalue": adf_level.get("p_value"),
            "adf_level_stationary": adf_level.get("is_stationary"),
            "adf_diff_stat": adf_diff.get("adf_statistic"),
            "adf_diff_pvalue": adf_diff.get("p_value"),
            "adf_diff_stationary": adf_diff.get("is_stationary"),
        }

        if acf_df is not None and not acf_df.empty:
            tested = acf_df[acf_df["lag"] >= 1]
            n_sig = int(tested["significant"].sum())
            lag1_p = tested.loc[tested["lag"] == 1, "ljung_box_pvalue"]
            lag1_p = float(lag1_p.iloc[0]) if not lag1_p.empty else None
            row["acf_n_significant_lags"] = n_sig
            row["acf_lags_tested"] = len(tested)
            row["acf_ljungbox_lag1_pvalue"] = lag1_p
            row["acf_has_autocorrelation"] = (lag1_p is not None) and (lag1_p <= alpha)
        else:
            row["acf_n_significant_lags"] = None
            row["acf_lags_tested"] = None
            row["acf_ljungbox_lag1_pvalue"] = None
            row["acf_has_autocorrelation"] = None

        rows.append(row)

    summary = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "test_summary.csv")
    summary.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("ADF & ACF TEST SUMMARY (all indicators)")
    print("=" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary.to_string(index=False))
    print(f"\nSaved: {csv_path}")

    # --- companion bar chart of key p-values vs alpha ---
    plot_rows = summary.dropna(subset=["adf_level_pvalue"])
    if not plot_rows.empty:
        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(plot_rows)), 5))
        x = np.arange(len(plot_rows))
        width = 0.25

        ax.bar(x - width, plot_rows["adf_level_pvalue"], width, label="ADF (level) p-value", color="#888888")
        ax.bar(x, plot_rows["adf_diff_pvalue"], width, label="ADF (first diff) p-value", color="#2a6fdb")
        ax.bar(x + width, plot_rows["acf_ljungbox_lag1_pvalue"], width,
               label="Ljung-Box (lag 1) p-value", color="#d62728")

        ax.axhline(alpha, color="black", linestyle="--", linewidth=1,
                    label=f"alpha = {alpha}")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_rows["indicator"], rotation=20, ha="right")
        ax.set_ylabel("p-value")
        ax.set_title("ADF & ACF (Ljung-Box) Test P-Values by Indicator")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plot_path = os.path.join(outdir, "test_summary_pvalues.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved: {plot_path}")

    return summary


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not INDICATORS:
        raise ValueError("INDICATORS is empty — add at least one series to the config at the top.")

    all_weekly = {}
    all_results = {}
    for name, cfg in INDICATORS.items():
        result = run_pipeline_for_indicator(name, cfg["path"], cfg["label"], OUTPUT_DIR)
        all_results[name] = result
        all_weekly[name] = result["weekly"]

    print("\n" + "#" * 70)
    print("# CROSS-INDICATOR SCATTERPLOTS")
    print("#" * 70)
    for x_name, y_name in SCATTER_PAIRS:
        plot_cross_indicator_scatter(all_weekly, x_name, y_name, OUTPUT_DIR)

    print("\n" + "#" * 70)
    print("# INDICATOR DIFFERENTIALS")
    print("#" * 70)
    for name1, name2 in DIFFERENTIAL_PAIRS:
        compute_indicator_differential(all_weekly, name1, name2, OUTPUT_DIR)

    build_test_summary(all_results, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print(f"Done. Processed {len(all_weekly)} indicator(s): {list(all_weekly.keys())}")
    print(f"All outputs saved in: {OUTPUT_DIR}/")
    print("=" * 70)


