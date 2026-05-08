# Thesis Factor Replication Analysis
## JKP Repo → 9 Factors for Volatility-Managed Portfolios Thesis

**Author:** Tomáš Šamaj, MSc Quantitative Finance, WU Vienna  
**Objective:** Replicate 9 factors (US stocks) at individual stock level to compute net trades across factors under volatility-scaling, following DeMiguel et al. (2024).

---

## 1. Repo Structure Overview

### Source files

| File | Role |
|---|---|
| `src/jkp/data/main.py` | Pipeline orchestrator — calls 35+ functions from aux_functions in sequence |
| `src/jkp/data/aux_functions.py` | Core library — all characteristic calculations, data transformations, I/O (~9000 lines) |
| `src/jkp/data/portfolio.py` | Factor portfolio construction — reads characteristics parquet, builds long-short factors via ECDF/tercile sorts |
| `src/jkp/data/config.py` | Constants (END_DATE, MAIN_FILTERS, PORTFOLIO_PFS=3, etc.) |
| `src/jkp/data/paths.py` | Path management (raw/, interim/, processed/) |
| `src/jkp/data/cli.py` | CLI entry points: `jkp build`, `jkp portfolio`, `jkp connect` |
| `src/jkp/data/wrds_credentials.py` | WRDS credential management (keyring-based) |
| `src/jkp/data/output_writer.py` | Write helpers for final outputs |

### Data flow

```
WRDS (26 tables) → raw/raw_tables/ → gen_raw_data_dfs() → interim/ (20+ parquet files)
→ 30+ processing functions → interim/world_data_output.parquet
→ processed/characteristics/{country}.parquet   (one file per country, all ~400 chars)
→ portfolio.py → processed/portfolios/lms.parquet, pfs.parquet, etc.
```

### Key intermediate files in interim/

| File | Content |
|---|---|
| `world_msf.parquet` | Merged monthly CRSP+Compustat stock file |
| `world_dsf.parquet` | Daily stock file |
| `acc_chars_world.parquet` | Accounting characteristics (annual+quarterly merged) |
| `market_chars_m.parquet` | Market characteristics (me, ret_12_1, etc.) |
| `world_data_prelim.parquet` | Merged panel (stock + accounting + market chars) |
| `ap_factors_monthly.parquet` | HXZ/FF factor returns time series (mktrf, hml, smb_ff, roe, inv, smb_hxz) |
| `roll_apply_daily.parquet` | All rolling daily characteristics (rvol_252d, corr_1260d, etc.) |
| `world_data_output.parquet` | Final filtered panel → split into country parquets |

---

## 2. The Nine Factors — Classification and Implementation Plan

### Group A — Direct use from portfolio.py (JKP construction matches target exactly)

| Factor | JKP characteristic column | Direction | Notes |
|---|---|---|---|
| **BAB** | `betabab_1260d` | -1 (short high-beta) | Computed as `corr_1260d × rvol_252d / mktvol_252d` in `finish_daily_chars()` — needs FULL daily pipeline |
| **ROE** | `niq_be` (quarterly net income / book equity) | +1 | JKP applies FF 2x3 sort on niq_be (sort_ff_style in ap_factors). The portfolio.py univariate sort on niq_be approximates HXZ triple 2x3x3 sort with high correlation. `niq_be` is in the acc_chars_list() and produced by `create_acc_chars("acc_std_qtr.parquet", ...)`. |
| **IA** | `at_gr1` (asset growth 1yr) | -1 (short high-investment) | Same note: portfolio.py univariate sort on at_gr1 approximates the HXZ inv factor. `at_gr1` is in acc_chars_list() and produced by `create_acc_chars("acc_std_ann.parquet", ...)`. |

> **Important note on ROE/IA:** The `ap_factors_monthly.parquet` file produced by `ap_factors()` in main.py contains factor-level return time series (`roe`, `inv`) computed with FF 2x3 sort on `niq_be`/`at_gr1`. These are **portfolio-level returns**, not stock-level weights. For stock-level weights that match the HXZ construction, you would need to extract weights from inside `sort_ff_style()`. The PDF instruction "run portfolio.py as-is" means: use portfolio.py's ECDF tercile sort on `niq_be` (for ROE) and `at_gr1` (for IA) applied to the characteristics parquet.

### Group B — Same characteristic, different sort (implement FF 2x3 in thesis_factors.py)

| Factor | JKP characteristic column | Required sort | Key difference from JKP |
|---|---|---|---|
| **MOM (UMD)** | `ret_12_1` (from `market_chars_monthly`) | FF 2x3 independent sort: NYSE median (size) × NYSE 30/70 (ret_12_2). Monthly rebalancing. | JKP uses ret_12_1 (t-12 to t-1); FF uses t-12 to t-2 (skip 1 month). Must recompute ret_12_2 from ret_exc in thesis_factors.py |
| **RMW** | `ope_be` (operating profit / book equity) | FF 2x3 independent sort: NYSE median (size) × NYSE 30/70 (ope_be). Annual June-end rebalancing. | JKP: univariate tercile monthly. FF: double-sort annual. Use ope_be directly. |
| **CMA** | `at_gr1` (asset growth 1yr) | FF 2x3 independent sort: NYSE median (size) × NYSE 30/70 (at_gr1). Annual June-end rebalancing. | JKP: univariate tercile monthly. FF: double-sort annual. Use at_gr1 directly. |

### Group C — Construct from raw CRSP/Compustat characteristics (implement in thesis_factors.py)

| Factor | Raw variables needed | Construction | Stock-level weight formula |
|---|---|---|---|
| **MktRF** | `me`, `ret_exc` | Value-weighted portfolio of all US stocks. No sorting. Monthly. | `w = ME_i / Σ ME_j` |
| **SMB** | `me` | FF 2x3 double sort on size × be_me (same sort as HML). SMB = avg(3 small portfolios) − avg(3 big portfolios). Annual June-end rebalancing. | Small leg: +1/3 per small pf (value-wtd within). Big leg: −1/3 per big pf. |
| **HML** | `me`, `be_me`, `be` | FF 2x3 independent sort: NYSE median (size) × NYSE 30/70 (B/M). Annual June-end rebalancing. Long high-B/M / short low-B/M. | High B/M: +1/2 across 2 high-B/M pfs. Low B/M: −1/2 across 2 low-B/M pfs. |

---

## 3. Pipeline Step Analysis — What Stays, What Can Be Cut

### 3A. Download step — `download_raw_data_tables()`

**26 tables downloaded from WRDS.** For US-only replication:

| Table | Keep? | Reason |
|---|---|---|
| `crsp.msf_v2` | **KEEP** | Core monthly stock data — me, ret, shares, etc. |
| `crsp.dsf_v2` | **KEEP** | Core daily stock data — needed for betabab_1260d |
| `comp.funda` | **KEEP** | Annual Compustat — at_gr1, ope_be, be_me, be |
| `comp.fundq` | **KEEP** | Quarterly Compustat — niq_be (quarterly ROE) |
| `crsp.ccmxpf_lnkhist` | **KEEP** | CRSP-Compustat link table |
| `comp.security` | **KEEP** | Security identifiers |
| `crsp.stksecurityinfohist` | **KEEP** | Stock security info |
| `crsp.stkissuerinfohist` | **KEEP** | Issuer info |
| `crsp.stkdelists` | **KEEP** | Delisting returns |
| `comp.company` | **KEEP** | Company header info |
| `comp.sec_history` | **KEEP** | Security history |
| `comp.co_hgic` | **KEEP** | GICS codes |
| `comp.r_ex_codes` | **KEEP** | Exchange codes |
| `comp.secm` | **KEEP** | Compustat monthly prices |
| `comp.secd` | **KEEP** | Compustat daily prices |
| `ff.factors_monthly` | **CAN DROP** | Used only for residual momentum (dropped) |
| `comp.exrt_dly` | **CAN DROP** | Currency exchange rates — not needed for US-only |
| `comp.g_security` | **CAN DROP** | Global Compustat — US only |
| `comp.g_company` | **CAN DROP** | Global Compustat — US only |
| `comp.g_sec_history` | **CAN DROP** | Global Compustat — US only |
| `comp.g_co_hgic` | **CAN DROP** | Global Compustat — US only |
| `comp.g_funda` | **CAN DROP** | Global Compustat — US only |
| `comp.g_fundq` | **CAN DROP** | Global Compustat — US only |
| `comp.g_secd` | **CAN DROP** | Global Compustat — US only |
| `crsp.indmthseriesdata_ind` | **CAN DROP** | CRSP industry index series — not needed |
| `crsp.indseriesinfohdr_ind` | **CAN DROP** | CRSP industry index header — not needed |

> **Caution:** gen_raw_data_dfs() and other early functions may fail if certain tables are missing. Safest approach: keep all downloads, drop at the output level. Cut downloads only after verifying the code path.

### 3B. main.py pipeline steps

| Step | Function | Keep? | Reason |
|---|---|---|---|
| 1 | `gen_raw_data_dfs()` | **KEEP** | Generates core intermediate files |
| 2 | `prepare_comp_sf("both")` | **KEEP** | Prepares Compustat security panel |
| 3 | `prepare_crsp_sf("m")` | **KEEP** | Monthly CRSP stock file — me, ret, etc. |
| 4 | `prepare_crsp_sf("d")` | **KEEP** | Daily CRSP — needed for betabab_1260d |
| 5 | `combine_crsp_comp_sf()` | **KEEP** | Merges CRSP and Compustat |
| 6 | `crsp_industry()` | **KEEP** | Industry codes needed for stock filtering flags |
| 7 | `comp_industry()` | **KEEP** | Industry codes |
| 8 | `merge_industry_to_world_msf()` | **KEEP** | Adds industry to stock file |
| 9 | `ff_ind_class()` | **KEEP** | FF49 industry classification (needed for ff49 filter flag) |
| 10 | `nyse_size_cutoffs()` | **KEEP** | **CRITICAL** — NYSE size breakpoints for ALL FF sorts |
| 11 | `classify_stocks_size_groups()` | **KEEP** | Size groups (mega/large/small/micro) for breakpoints |
| 12 | `return_cutoffs("m", 0)` | **KEEP** | Monthly return winsorization |
| 13 | `return_cutoffs("d", 0)` | **KEEP** | Daily return winsorization (for BAB) |
| 14 | `add_ret_exc_wins("m")` | **KEEP** | Add winsorized excess returns |
| 15 | `add_ret_exc_wins("d")` | **KEEP** | Daily winsorized returns |
| 16 | `market_returns("d", ...)` | **KEEP** | Daily market returns (needed for corr_1260d) |
| 17 | `market_returns("m", ...)` | **KEEP** | Monthly market returns (me, market index) |
| 18 | `standardized_accounting_data(...)` | **KEEP** | Builds accounting panel. Internally computes many intermediates; final output contains at_gr1, ope_be, be_me, be, niq_be among others. |
| 19 | `create_acc_chars("acc_std_ann.parquet", ...)` | **KEEP** | Annual accounting chars including at_gr1, ope_be, be_me, be |
| 20 | `create_acc_chars("acc_std_qtr.parquet", ...)` | **KEEP** | Quarterly chars including niq_be |
| 21 | `combine_ann_qtr_chars(...)` | **KEEP** | Merges ann/qtr (uses fresher quarterly where available) |
| 22 | `market_chars_monthly(...)` | **KEEP** | Computes ret_12_1, market_equity (me), dividends, etc. |
| 23 | `create_world_data_prelim(...)` | **KEEP** | Assembles main panel |
| 24 | `ap_factors("d", ...)` | **CAN DROP** | Daily HXZ factors — only needed if residual_momentum uses daily (it uses monthly) |
| 25 | `ap_factors("m", ...)` | **CAN DROP** | Monthly HXZ factors — only needed for residual_momentum (dropped) and factor regressions (dropped). However it writes ap_factors_monthly.parquet which is copied to other_output/. If thesis_factors.py needs it, KEEP. |
| 26 | `firm_age(...)` | **DROP** | Not needed for any of the 9 factors |
| 27 | `mispricing_factors(...)` | **DROP** | Not needed |
| 28 | `market_beta("beta_60m.parquet", ..., 60, 36)` | **DROP** | 60m rolling beta — not the same as betabab_1260d |
| 29 | `residual_momentum(...)` × 2 | **DROP** | Not needed |
| 30 | `bidask_hl(...)` | **DROP** | Corwin-Schultz bid-ask — not needed |
| 31 | `prepare_daily(...)` | **KEEP** | Prepares daily data for roll_apply_daily |
| 32 | `roll_apply_daily("rvol", "_21d", 15)` | **DROP** | 21d volatility — not needed |
| 33 | `roll_apply_daily("rmax", "_21d", 15)` | **DROP** | Max return — not needed |
| 34 | `roll_apply_daily("skew", "_21d", 15)` | **DROP** | Skewness — not needed |
| 35 | `roll_apply_daily("capm_ext", "_21d", 15)` | **DROP** | CAPM extension — not needed |
| 36 | `roll_apply_daily("ff3", "_21d", 15)` | **DROP** | FF3 — not needed |
| 37 | `roll_apply_daily("hxz4", "_21d", 15)` | **DROP** | HXZ4 — not needed |
| 38 | `roll_apply_daily("dimsonbeta", "_21d", 15)` | **DROP** | Dimson beta — not needed |
| 39 | `roll_apply_daily("zero_trades", "_21d", 15)` | **DROP** | Not needed |
| 40 | `roll_apply_daily("zero_trades", "_126d", 60)` | **DROP** | Not needed |
| 41 | `roll_apply_daily("turnover", "_126d", 60)` | **DROP** | Not needed |
| 42 | `roll_apply_daily("dolvol", "_126d", 60)` | **DROP** | Not needed |
| 43 | `roll_apply_daily("ami", "_126d", 60)` | **DROP** | Not needed |
| 44 | **`roll_apply_daily("rvol", "_252d", 120)`** | **KEEP** | **rvol_252d — needed for betabab_1260d** |
| 45 | `roll_apply_daily("capm", "_252d", 120)` | **DROP** | CAPM beta — not needed |
| 46 | `roll_apply_daily("downbeta", "_252d", 120)` | **DROP** | Not needed |
| 47 | `roll_apply_daily("zero_trades", "_252d", 120)` | **DROP** | Not needed |
| 48 | `roll_apply_daily("prc_to_high", "_252d", 120)` | **DROP** | Not needed |
| 49 | **`roll_apply_daily("mktvol", "_252d", 120)`** | **KEEP** | **mktvol_252d — denominator in betabab_1260d** |
| 50 | **`roll_apply_daily("mktcorr", "_1260d", 750)`** | **KEEP** | **corr_1260d — numerator correlation in betabab_1260d** |
| 51 | `merge_roll_apply_daily_results()` | **KEEP** | Assembles daily chars parquet |
| 52 | `finish_daily_chars(...)` | **KEEP** | Computes `betabab_1260d = corr_1260d × rvol_252d / mktvol_252d` |
| 53 | `merge_world_data_prelim()` | **KEEP (simplified)** | Merges: world_data_prelim + market_chars_d + beta_60m + resmom + mispricing + firm_age. After dropping the dropped steps, only market_chars_d and the core prelim are needed. Remove joins to dropped files. |
| 54 | `quality_minus_junk(...)` | **DROP** | QMJ factor — not needed |
| 55 | `merge_qmj_to_world_data()` | **DROP** | QMJ — not needed |
| 56 | `filter_dsf()` | **DROP** | Daily stock filter output — daily data for BAB is already captured before this |
| 57 | `filter_msf()` | **KEEP** | Monthly filter (but simplify to drop daily output) |
| 58 | `filter_world()` | **KEEP** | Applies MAIN_FILTERS to produce world_data_output.parquet |
| 59 | `save_main_data(paths)` | **KEEP** | Writes per-country characteristics parquets |
| 60 | `save_daily_ret()` | **KEEP** | Saves daily returns by country (needed for portfolio.py daily returns) |
| 61 | `save_monthly_ret()` | **KEEP** | Saves monthly returns |
| 62 | `save_accounting_data()` | **CAN DROP** | Saves acc panel separately — not needed for thesis |
| 63 | `save_output_files()` | **KEEP (simplified)** | Copies cutoff files, market returns, nyse_cutoffs to processed/other_output/ |
| 64 | `save_full_files_and_cleanup(clear_interim=True)` | **KEEP** | Cleanup |

### 3C. acc_chars_list() — what to keep

The full list has ~250 characteristics. The `standardized_accounting_data()` function computes ALL of them internally (they share intermediate variables). The simplification is to reduce what gets **saved to disk**, not necessarily what's computed:

**Accounting characteristics needed (keep in acc_chars_list):**
- `at_gr1` — asset growth 1yr (for CMA and IA/Group A)
- `ope_be` — operating profits / book equity (for RMW)
- `be_me` — book-to-market equity (for HML, SMB)
- `be` — book equity (computed internally; ensure it's retained as `book_equity` or `be_me` numerator)
- `niq_be` — quarterly net income / book equity (for ROE/Group A)

> **Warning:** `standardized_accounting_data()` uses many intermediate columns that are dependencies for others. Truncating acc_chars_list() to only 5 items should be safe because the output filter happens at the end of the function via `rename_cols_and_select_keep_vars()`. However, some complex characteristics (like equity duration `eq_dur`, f_score, z_score) call helper functions that must run in the right order. Since we're NOT computing these, their helper functions would be skipped. **Risk:** some helper functions might modify shared state (the chars DataFrame) in ways that affect subsequent calculations. This needs careful testing.

**Safe approach:** Start by keeping the full acc_chars_list() in the first version, then prune after verifying correctness.

### 3D. portfolio.py simplifications

For Group A factors only:

**Keep:**
- `portfolios()` function, but restricted to `chars = ["betabab_1260d", "niq_be", "at_gr1"]`
- Only for `excntry = "usa"`
- `bps = "nyse"` (for consistency with FF sorts)
- The `hml_returns` and `lms_returns` computation

**Drop:**
- All regional portfolio construction (`regional_data()`, regional loops)
- All cluster portfolio construction
- CMP portfolio construction (`cmp_key=True` path)
- Daily portfolio construction (`daily_pf=True`)
- Industry portfolio construction (`ind_pf=True`)
- Signals output (`signals=True`)
- All writing of regional/cluster/country subdirectory parquets
- All non-USA countries

---

## 4. thesis_factors.py — New Module to Create

This module reads the stock-level panel output from main.py and constructs stock-level weights for the 6 non-Group-A factors. Should be placed at `src/jkp/data/thesis_factors.py`.

### Inputs (from main.py output in `processed/characteristics/USA.parquet`)

| Column | Used for |
|---|---|
| `id` | Stock identifier (permno for CRSP) |
| `eom` | Month-end date |
| `excntry` | Country filter (keep == "USA") |
| `me` | Market equity — MktRF weights; size breakpoints |
| `be_me` | Book-to-market — HML sort; SMB auxiliary |
| `be` | Book equity — needed alongside be_me for consistency |
| `ope_be` | Operating profit / BE — RMW sort |
| `at_gr1` | Asset growth — CMA sort |
| `ret_12_1` | Prior 12m return (t-12 to t-1) — reconstruct as ret_12_2 for MOM |
| `ret_exc` | Excess stock return — for ret_12_2 reconstruction and all factor return calculations |
| `crsp_exchcd` | Exchange code — NYSE stocks (exchcd==1) for breakpoints |
| `comp_exchg` | Compustat exchange code — also for NYSE identification |
| `common` | Common stock indicator — stock filter |
| `primary_sec` | Primary security indicator |
| `obs_main` | Observation main filter |
| `exch_main` | Exchange main filter |
| `size_grp` | Size group (mega/large/small/micro) |
| `source_crsp` | Source flag (1=CRSP, 0=Compustat) |

### Required also from `processed/other_output/nyse_cutoffs.parquet`

- `nyse_p20`, `nyse_p50` (median for size split), `nyse_p30`, `nyse_p70` (30/70 for char split)

### Core functions to implement

```python
def ff_double_sort(df, char_col, rebalance='annual'):
    """
    FF 2x3 independent double sort on size x char_col.
    
    1. At rebalancing date (June for annual, every month for monthly):
       a. NYSE median of me → size breakpoint (small/big)
       b. NYSE 30th/70th percentile of char_col → char breakpoints (low/mid/high)
       c. Assign each stock to 2x3 = 6 portfolios
    2. Hold from July(t) to June(t+1) for annual; 1 month for monthly
    3. Value-weight within each portfolio using me
    4. Return stock-level weights w_{i,t} with sign applied
    
    Returns: DataFrame with (date, id, portfolio_label, weight_long, weight_short)
    """

def compute_mktrf(df):
    """Value-weighted portfolio of all stocks. w = ME_i / sum(ME)."""

def compute_hml_smb(df, nyse_cutoffs):
    """
    FF 2x3 sort on be_me. Annual June rebalancing.
    HML = high-B/M − low-B/M (½ × (small-H + big-H) − ½ × (small-L + big-L))
    SMB_HML = ⅓ × (small-H + small-M + small-L) − ⅓ × (big-H + big-M + big-L)
    """

def compute_rmw(df, nyse_cutoffs):
    """FF 2x3 sort on ope_be. Annual June rebalancing."""

def compute_cma(df, nyse_cutoffs):
    """FF 2x3 sort on at_gr1. Annual June rebalancing."""

def compute_mom(df, nyse_cutoffs):
    """
    FF 2x3 sort on ret_12_2 (skip-month corrected). Monthly rebalancing.
    ret_12_2 = cumulative return t-12 to t-2 (excludes month t-1).
    Reconstructed from ret_exc column: ret_12_2 = ret_12_1 / (1 + ret_exc_lag1) − 1
    """

def run_thesis_factors(data_dir):
    """
    Main entry: reads characteristics, calls all factor functions,
    joins with Group A weights from portfolio.py output,
    writes thesis_factor_weights.parquet.
    """
```

### Output format

`data/processed/thesis_factor_weights.parquet`

| Column | Type | Description |
|---|---|---|
| `date` | Date | Month-end date |
| `permno` | Int | Stock identifier |
| `w_MktRF` | Float | Weight in MktRF factor |
| `w_SMB` | Float | Weight in SMB factor |
| `w_HML` | Float | Weight in HML factor |
| `w_MOM` | Float | Weight in MOM (UMD) factor |
| `w_RMW` | Float | Weight in RMW factor |
| `w_CMA` | Float | Weight in CMA factor |
| `w_ROE` | Float | Weight in ROE factor (from portfolio.py JKP sort) |
| `w_IA` | Float | Weight in IA factor (from portfolio.py JKP sort) |
| `w_BAB` | Float | Weight in BAB factor (from portfolio.py JKP sort) |

Long leg: sum of weights = +1. Short leg: sum = -1. One row per stock-month with non-zero weight in at least one factor.

---

## 5. Implementation Plan (Sequential Steps)

### Phase 1 — Run existing pipeline, extract Group A (no code changes)

1. Run `jkp build data/` → generates characteristics
2. Simplify `portfolio.py` to US-only, 3 chars: `betabab_1260d`, `niq_be`, `at_gr1`
3. Run `jkp portfolio data/` → generates pfs.parquet and lms.parquet
4. Validate Group A factor returns vs. benchmarks (AQR for BAB, q-factor database for ROE/IA)

### Phase 2 — Simplify main.py pipeline (drop unused steps)

1. Remove `firm_age()`, `mispricing_factors()`, `market_beta()`, `residual_momentum()` × 2, `bidask_hl()`, `quality_minus_junk()`, `merge_qmj_to_world_data()`, `ap_factors()` daily
2. Remove unnecessary `roll_apply_daily()` calls (keep only rvol_252d, mktvol_252d, mktcorr_1260d)
3. Simplify `merge_world_data_prelim()` to not join beta_60m, resmom, mp_factors, firm_age
4. Simplify `acc_chars_list()` to output only needed columns (at_gr1, ope_be, be_me, niq_be + a few base vars)
5. Run tests: `uv run pytest tests/unit/` — expect some failures (tests for removed characteristics)
6. Validate that remaining 9-factor chars are still correct

### Phase 3 — Create thesis_factors.py (Group B and C)

1. Implement `ff_double_sort()` helper
2. Implement MOM (with ret_12_2 skip-month correction)
3. Implement RMW and CMA (annual June rebalancing)
4. Implement MktRF (simple value-weighted)
5. Implement HML and SMB (standard FF 2x3)
6. Validate each factor vs. Kenneth French data library (target correlation > 0.97)

### Phase 4 — Assemble output and validate

1. Join Group A weights (from portfolio.py) with Group B/C weights (from thesis_factors.py)
2. Write `thesis_factor_weights.parquet`
3. Validate all 9 factor returns against benchmarks
4. Confirm output format matches DeMiguel et al. netting formula requirements

---

## 6. Key Risks and Decisions

### Risk 1: ROE/IA — HXZ triple sort vs. JKP univariate
The exact HXZ construction uses a triple 2×3×3 sort (size × investment × ROE), which JKP approximates with a 2×3 FF-style sort inside `ap_factors()`. Portfolio.py's univariate sort on `niq_be` and `at_gr1` is a further simplification. **Decision:** Start with portfolio.py univariate sort and validate correlation vs. q-factor database. If < 0.97, implement the FF 2×3 sort from `sort_ff_style()` logic in thesis_factors.py.

### Risk 2: acc_chars_list() pruning breaking intermediate calculations
Some characteristics in acc_chars_list() depend on intermediate variables computed for other characteristics. Prune the output list last (after verifying all required outputs are correct), not the intermediate computations.

### Risk 3: betabab_1260d requires 5 years of daily data (1260 trading days)
Need sufficient historical data. `mktcorr_1260d` requires `__min=750` trading days. Ensure date range is set early enough (default END_DATE = 2025-12-31 is fine, just need data from 2003 forward to get 1260-day windows by 2008).

### Risk 4: NYSE breakpoints for FF sorts
The FF 2×3 sorts require NYSE stocks ONLY for computing the size median and char 30/70 breakpoints, then applying those breakpoints to ALL stocks (NYSE + AMEX + NASDAQ). The `nyse_cutoffs.parquet` file provides the size breakpoints. For char breakpoints, thesis_factors.py must filter to NYSE stocks (crsp_exchcd==1 or comp_exchg==11) when computing percentiles.

### Risk 5: prepare_daily() dependency on ap_factors_daily
`prepare_daily("world_dsf.parquet", "ap_factors_daily.parquet")` — if we drop `ap_factors("d", ...)`, this call will fail. Two options: (a) keep ap_factors daily even though we don't use HXZ daily factors, or (b) check if prepare_daily strictly needs ap_factors_daily and if we can pass a dummy/empty frame for the vars we're actually computing (rvol_252d, mktvol_252d, mktcorr_1260d).

### Risk 6: save_main_data() uses shell commands (bash for-loop) — Windows incompatible
The `save_main_data()` function uses `os.system()` with bash shell commands to rename partitioned parquet files. This will fail on Windows. This is already a known limitation of the repo when running on Windows vs. Linux. When running on a WRDS server or Linux cluster, this is fine. For local Windows testing, may need a Python alternative.

---

## 7. Summary: What Changes Are Needed

### Files to KEEP unchanged (or nearly so):
- `config.py`, `paths.py`, `cli.py`, `wrds_credentials.py`, `output_writer.py`
- `aux_functions.py` — main body unchanged; only drop calls to unused functions from `main.py`

### Files to MODIFY:
- `main.py` — remove calls to ~15 dropped functions; simplify `merge_world_data_prelim()` join list
- `portfolio.py` — restrict to USA only, 3 chars, drop regional/cluster/CMP/daily/industry outputs

### Files to CREATE:
- `src/jkp/data/thesis_factors.py` — new module implementing FF double-sort engine and all 6 Group B/C factor constructions

### Tests:
- Most existing unit tests remain valid (they test expression-level helpers, not the full pipeline)
- Some tests for dropped characteristics may need to be removed or skipped
- Add new tests in `tests/unit/test_thesis_factors.py` for the new ff_double_sort logic

### Resource files (unchanged):
- `resources/factor_details.xlsx` — still needed by portfolio.py (char info + directions)
- `resources/country_classification.xlsx` — still needed (even for USA-only, it provides the framework)
- `resources/cluster_labels.csv` — can be dropped after simplifying portfolio.py
- `resources/Siccodes49.txt` — still needed for FF49 classification (used in ff_ind_class)
