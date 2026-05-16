"""
Stock-level factor weight construction for all 9 thesis factors.

Factors and their construction:
  Group A — univariate tercile sort, non-microcap breakpoints (matches JKP portfolio.py):
    BAB  betabab_1260d  direction=-1  (short high-beta)
    ROE  niq_be         direction=+1  (long high-ROE)
    IA   at_gr1         direction=-1  (short high-investment)

  Group B — FF 2×3 independent double sort, NYSE breakpoints:
    RMW  ope_be   annual June  long=Robust(H), short=Weak(L)
    CMA  at_gr1   annual June  long=Conservative(L), short=Aggressive(H)
    MOM  ret_12_2 monthly      long=Winner(H), short=Loser(L)
      ret_12_2 = (1+ret_12_1)/(1+ret_exc_lag1) - 1  (skip-month correction)

  Group C — FF sorts + value-weighted all stocks:
    HML   be_me  annual June  ½(S/H + B/H) - ½(S/L + B/L)
    SMB   FF5: average of SMBs from be_me, ope_be, at_gr1 sorts (Fama-French 2015)
    MktRF value-weighted all stocks, w = me_i / Σme

Output: processed/thesis_factor_weights.parquet
  Columns: eom, id, w_MktRF, w_SMB, w_HML, w_MOM, w_RMW, w_CMA, w_ROE, w_IA, w_BAB
  Long leg: weights sum to +1. Short leg: weights sum to -1.
  One row per (eom, id) with at least one non-zero weight.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAIN_FILTER = (
    (pl.col("primary_sec") == 1)
    & (pl.col("common") == 1)
    & (pl.col("obs_main") == 1)
    & (pl.col("exch_main") == 1)
    & (pl.col("me") > 0)
)

# NYSE stock identification (matches portfolio.py bps="nyse" logic)
_NYSE = (
    ((pl.col("crsp_exchcd") == 1) & pl.col("comp_exchg").is_null())
    | ((pl.col("comp_exchg") == 11) & pl.col("crsp_exchcd").is_null())
)

_WEIGHT_COLS = ["w_MktRF", "w_SMB", "w_HML", "w_MOM", "w_RMW", "w_CMA", "w_ROE", "w_IA", "w_BAB"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reb_year() -> pl.Expr:
    """Map eom to its June rebalancing year: Jul-Dec → same year; Jan-Jun → prior year."""
    return (
        pl.when(pl.col("eom").dt.month() >= 7)
        .then(pl.col("eom").dt.year())
        .otherwise(pl.col("eom").dt.year() - 1)
    )


def _june_breakpoints(df: pl.DataFrame, char_col: str, *, positive_char: bool = False) -> pl.DataFrame:
    """
    Annual June NYSE breakpoints for size (p50) and char (p30 / p70).
    Returns DataFrame with columns [reb_yr, me_bp50, char_bp30, char_bp70].
    """
    char_filter = pl.col(char_col).is_not_null()
    if positive_char:
        char_filter = char_filter & (pl.col(char_col) > 0)

    return (
        df.filter((pl.col("eom").dt.month() == 6) & _NYSE & (pl.col("me") > 0) & char_filter)
        .group_by("eom")
        .agg([
            pl.col("me").quantile(0.5, interpolation="linear").alias("me_bp50"),
            pl.col(char_col).quantile(0.3, interpolation="linear").alias("char_bp30"),
            pl.col(char_col).quantile(0.7, interpolation="linear").alias("char_bp70"),
        ])
        .with_columns(pl.col("eom").dt.year().alias("reb_yr"))
        .drop("eom")
    )


def _monthly_breakpoints(df: pl.DataFrame, char_col: str) -> pl.DataFrame:
    """Monthly NYSE breakpoints for size (p50) and char (p30 / p70)."""
    return (
        df.filter(_NYSE & (pl.col("me") > 0) & pl.col(char_col).is_not_null())
        .group_by("eom")
        .agg([
            pl.col("me").quantile(0.5, interpolation="linear").alias("me_bp50"),
            pl.col(char_col).quantile(0.3, interpolation="linear").alias("char_bp30"),
            pl.col(char_col).quantile(0.7, interpolation="linear").alias("char_bp70"),
        ])
    )


def _assign_portfolios(
    df: pl.DataFrame,
    bps: pl.DataFrame,
    char_col: str,
    join_col: str,
) -> pl.DataFrame:
    """
    Join breakpoints and assign sz_pf ∈ {S, B} and char_pf ∈ {L, M, H}.
    join_col: 'eom' for monthly, 'reb_yr' for annual.
    """
    return (
        df.join(bps, on=join_col, how="left")
        .with_columns([
            pl.when(pl.col("me") < pl.col("me_bp50"))
            .then(pl.lit("S"))
            .otherwise(pl.lit("B"))
            .alias("sz_pf"),
            pl.when(pl.col(char_col).is_null())
            .then(None)
            .when(pl.col(char_col) <= pl.col("char_bp30"))
            .then(pl.lit("L"))
            .when(pl.col(char_col) >= pl.col("char_bp70"))
            .then(pl.lit("H"))
            .otherwise(pl.lit("M"))
            .alias("char_pf"),
        ])
        .drop(["me_bp50", "char_bp30", "char_bp70"])
    )


def _smb_from_sort(df: pl.DataFrame, char_col: str, *, positive_char: bool = False) -> pl.DataFrame:
    """
    Per-stock SMB weight contribution from one annual 2×3 sort.
    Returns (eom, id, w_smb) where w_smb = +(1/3)×vw for small, −(1/3)×vw for big.
    """
    df_sort = df.filter(pl.col(char_col).is_not_null())
    if positive_char:
        df_sort = df_sort.filter(pl.col(char_col) > 0)
    bps = _june_breakpoints(df_sort, char_col, positive_char=positive_char)
    return (
        df_sort.with_columns(_reb_year().alias("reb_yr"))
        .pipe(_assign_portfolios, bps, char_col, "reb_yr")
        .filter(pl.col("sz_pf").is_not_null() & pl.col("char_pf").is_not_null())
        .drop("reb_yr")
        .pipe(_value_weight, ["eom", "sz_pf", "char_pf"])
        .with_columns(
            pl.when(pl.col("sz_pf") == "S")
            .then((1 / 3) * pl.col("vw"))
            .otherwise(-(1 / 3) * pl.col("vw"))
            .alias("w_smb")
        )
        .select(["eom", "id", "w_smb"])
    )


def _value_weight(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    """Add 'vw' = me_i / Σme within group. Group must have me > 0."""
    return df.with_columns(
        (pl.col("me") / pl.col("me").sum().over(group_cols)).alias("vw")
    )


# ---------------------------------------------------------------------------
# Group C: MktRF, HML, SMB
# ---------------------------------------------------------------------------

def compute_mktrf(df: pl.DataFrame) -> pl.DataFrame:
    """Value-weighted portfolio of all stocks: w_MktRF = me_i / Σme per month."""
    return (
        df.with_columns(
            (pl.col("me") / pl.col("me").sum().over("eom")).alias("w_MktRF")
        )
        .select(["eom", "id", "w_MktRF"])
    )


def compute_hml_smb(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    HML: FF 2×3 sort on be_me (positive only), annual June rebalancing.
      HML = ½(S/H + B/H) − ½(S/L + B/L)
    SMB: FF5 construction (Fama-French 2015) — average of SMBs from three independent
      2×3 annual sorts on be_me, ope_be, and at_gr1.
    Returns (hml_weights, smb_weights) each with columns [eom, id, w_*].
    """
    df_bm = df.filter(pl.col("be_me").is_not_null() & (pl.col("be_me") > 0))
    bps_bm = _june_breakpoints(df_bm, "be_me", positive_char=True)
    assigned_bm = (
        df_bm.with_columns(_reb_year().alias("reb_yr"))
        .pipe(_assign_portfolios, bps_bm, "be_me", "reb_yr")
        .filter(pl.col("sz_pf").is_not_null() & pl.col("char_pf").is_not_null())
        .drop("reb_yr")
        .pipe(_value_weight, ["eom", "sz_pf", "char_pf"])
    )

    hml = (
        assigned_bm.with_columns(
            pl.when(pl.col("char_pf") == "H").then(0.5 * pl.col("vw"))
            .when(pl.col("char_pf") == "L").then(-0.5 * pl.col("vw"))
            .otherwise(0.0)
            .alias("w_HML")
        )
        .filter(pl.col("w_HML") != 0.0)
        .select(["eom", "id", "w_HML"])
    )

    # FF5 SMB: average the per-stock contributions from all three sorts
    smb_bm = _smb_from_sort(df, "be_me", positive_char=True).rename({"w_smb": "w_bm"})
    smb_op = _smb_from_sort(df, "ope_be").rename({"w_smb": "w_op"})
    smb_inv = _smb_from_sort(df, "at_gr1").rename({"w_smb": "w_inv"})
    smb = (
        smb_bm
        .join(smb_op, on=["eom", "id"], how="full", coalesce=True)
        .join(smb_inv, on=["eom", "id"], how="full", coalesce=True)
        .fill_null(0.0)
        .with_columns(pl.col("id").cast(pl.Int64))
        .with_columns(
            ((pl.col("w_bm") + pl.col("w_op") + pl.col("w_inv")) / 3).alias("w_SMB")
        )
        .filter(pl.col("w_SMB") != 0.0)
        .select(["eom", "id", "w_SMB"])
    )

    return hml, smb


# ---------------------------------------------------------------------------
# Group B: RMW, CMA, MOM
# ---------------------------------------------------------------------------

def compute_rmw(df: pl.DataFrame) -> pl.DataFrame:
    """
    FF 2×3 sort on ope_be. Annual June rebalancing.
    Long = Robust (high ope_be, char_pf=H), Short = Weak (low ope_be, char_pf=L).
    """
    df_sort = df.filter(pl.col("ope_be").is_not_null())
    bps = _june_breakpoints(df_sort, "ope_be")

    return (
        df_sort.with_columns(_reb_year().alias("reb_yr"))
        .pipe(_assign_portfolios, bps, "ope_be", "reb_yr")
        .filter(pl.col("sz_pf").is_not_null() & pl.col("char_pf").is_not_null())
        .drop("reb_yr")
        .pipe(_value_weight, ["eom", "sz_pf", "char_pf"])
        .with_columns(
            pl.when(pl.col("char_pf") == "H").then(0.5 * pl.col("vw"))
            .when(pl.col("char_pf") == "L").then(-0.5 * pl.col("vw"))
            .otherwise(0.0)
            .alias("w_RMW")
        )
        .filter(pl.col("w_RMW") != 0.0)
        .select(["eom", "id", "w_RMW"])
    )


def compute_cma(df: pl.DataFrame) -> pl.DataFrame:
    """
    FF 2×3 sort on at_gr1. Annual June rebalancing.
    Long = Conservative (low at_gr1, char_pf=L), Short = Aggressive (high at_gr1, char_pf=H).
    """
    df_sort = df.filter(pl.col("at_gr1").is_not_null())
    bps = _june_breakpoints(df_sort, "at_gr1")

    return (
        df_sort.with_columns(_reb_year().alias("reb_yr"))
        .pipe(_assign_portfolios, bps, "at_gr1", "reb_yr")
        .filter(pl.col("sz_pf").is_not_null() & pl.col("char_pf").is_not_null())
        .drop("reb_yr")
        .pipe(_value_weight, ["eom", "sz_pf", "char_pf"])
        .with_columns(
            # Low at_gr1 = conservative = long
            pl.when(pl.col("char_pf") == "L").then(0.5 * pl.col("vw"))
            .when(pl.col("char_pf") == "H").then(-0.5 * pl.col("vw"))
            .otherwise(0.0)
            .alias("w_CMA")
        )
        .filter(pl.col("w_CMA") != 0.0)
        .select(["eom", "id", "w_CMA"])
    )


def compute_mom(df: pl.DataFrame) -> pl.DataFrame:
    """
    FF 2×3 sort on ret_12_2 (skip-month momentum). Monthly rebalancing.
    ret_12_2 = (1 + ret_12_1) / (1 + ret_exc_{t-1}) − 1 drops the t-1 month.
    Long = Winner (high ret_12_2, char_pf=H), Short = Loser (low, char_pf=L).
    """
    df_mom = (
        df.sort(["id", "eom"])
        .with_columns(
            (
                (1 + pl.col("ret_12_1")) / (1 + pl.col("ret_exc").shift(1).over("id")) - 1
            ).alias("ret_12_2")
        )
        .filter(pl.col("ret_12_2").is_not_null())
    )

    bps = _monthly_breakpoints(df_mom, "ret_12_2")

    return (
        df_mom.pipe(_assign_portfolios, bps, "ret_12_2", "eom")
        .filter(pl.col("sz_pf").is_not_null() & pl.col("char_pf").is_not_null())
        .pipe(_value_weight, ["eom", "sz_pf", "char_pf"])
        .with_columns(
            pl.when(pl.col("char_pf") == "H").then(0.5 * pl.col("vw"))
            .when(pl.col("char_pf") == "L").then(-0.5 * pl.col("vw"))
            .otherwise(0.0)
            .alias("w_MOM")
        )
        .filter(pl.col("w_MOM") != 0.0)
        .select(["eom", "id", "w_MOM"])
    )


# ---------------------------------------------------------------------------
# Group A: BAB (Frazzini-Pedersen), ROE (q-factor 2×3), IA (q-factor 2×3 monthly)
# ---------------------------------------------------------------------------

def compute_roe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Q-factor ROE: FF 2×3 sort on niq_be. Annual June rebalancing.
    Long = high ROE (char_pf=H), Short = low ROE (char_pf=L).
    Matches Hou-Xue-Zhang q5 r_roe construction.
    """
    df_sort = df.filter(pl.col("niq_be").is_not_null())
    bps = _june_breakpoints(df_sort, "niq_be")

    return (
        df_sort.with_columns(_reb_year().alias("reb_yr"))
        .pipe(_assign_portfolios, bps, "niq_be", "reb_yr")
        .filter(pl.col("sz_pf").is_not_null() & pl.col("char_pf").is_not_null())
        .drop("reb_yr")
        .pipe(_value_weight, ["eom", "sz_pf", "char_pf"])
        .with_columns(
            pl.when(pl.col("char_pf") == "H").then(0.5 * pl.col("vw"))
            .when(pl.col("char_pf") == "L").then(-0.5 * pl.col("vw"))
            .otherwise(0.0)
            .alias("w_ROE")
        )
        .filter(pl.col("w_ROE") != 0.0)
        .select(["eom", "id", "w_ROE"])
    )


def compute_ia(df: pl.DataFrame) -> pl.DataFrame:
    """
    Q-factor IA: FF 2×3 sort on at_gr1. Monthly rebalancing (distinct from CMA's June-only).
    Long = low investment (char_pf=L), Short = high investment (char_pf=H).
    Matches Hou-Xue-Zhang q5 r_ia construction.
    """
    df_sort = df.filter(pl.col("at_gr1").is_not_null())
    bps = _monthly_breakpoints(df_sort, "at_gr1")

    return (
        df_sort.pipe(_assign_portfolios, bps, "at_gr1", "eom")
        .filter(pl.col("sz_pf").is_not_null() & pl.col("char_pf").is_not_null())
        .pipe(_value_weight, ["eom", "sz_pf", "char_pf"])
        .with_columns(
            pl.when(pl.col("char_pf") == "L").then(0.5 * pl.col("vw"))
            .when(pl.col("char_pf") == "H").then(-0.5 * pl.col("vw"))
            .otherwise(0.0)
            .alias("w_IA")
        )
        .filter(pl.col("w_IA") != 0.0)
        .select(["eom", "id", "w_IA"])
    )


def compute_bab(df: pl.DataFrame) -> pl.DataFrame:
    """
    Frazzini-Pedersen (2014) BAB: rank-weighted, each leg scaled to unit beta.
    Long = low-beta stocks (centered rank < 0); Short = high-beta stocks (centered rank > 0).
    Weights are rank-based (not value-weighted); the net beta of the factor is zero.
    """
    return (
        df.filter(pl.col("betabab_1260d").is_not_null())
        .with_columns(
            pl.col("betabab_1260d").rank("average").over("eom").alias("beta_rank"),
            pl.col("betabab_1260d").count().over("eom").alias("n"),
        )
        .with_columns(
            (pl.col("beta_rank") - (pl.col("n") + 1) / 2).alias("z")
        )
        .with_columns(
            pl.when(pl.col("z") < 0).then(pl.col("z").abs()).alias("z_long"),
            pl.when(pl.col("z") > 0).then(pl.col("z").abs()).alias("z_short"),
        )
        .with_columns(
            (pl.col("z_long") / pl.col("z_long").sum().over("eom")).alias("w_long_raw"),
            (pl.col("z_short") / pl.col("z_short").sum().over("eom")).alias("w_short_raw"),
        )
        .with_columns(
            (pl.col("w_long_raw") * pl.col("betabab_1260d")).sum().over("eom").alias("beta_L"),
            (pl.col("w_short_raw") * pl.col("betabab_1260d")).sum().over("eom").alias("beta_H"),
        )
        .with_columns(
            pl.when(pl.col("z") < 0).then(pl.col("w_long_raw") / pl.col("beta_L"))
            .when(pl.col("z") > 0).then(-(pl.col("w_short_raw") / pl.col("beta_H")))
            .otherwise(0.0)
            .alias("w_BAB")
        )
        .filter(pl.col("w_BAB") != 0.0)
        .select(["eom", "id", "w_BAB"])
    )


def _univariate_tercile(
    df: pl.DataFrame, char_col: str, direction: int, weight_name: str
) -> pl.DataFrame:
    """
    Univariate tercile sort using non-microcap (size_grp ∈ mega/large/small) breakpoints.
    Matches JKP portfolio.py Group A construction with bps='non_mc'.
    direction: +1 → long=pf3(high char), short=pf1; -1 → long=pf1(low char), short=pf3.
    """
    bps = (
        df.filter(
            pl.col("size_grp").is_in(["mega", "large", "small"])
            & pl.col(char_col).is_not_null()
        )
        .group_by("eom")
        .agg([
            pl.col(char_col).quantile(1 / 3, interpolation="linear").alias("p33"),
            pl.col(char_col).quantile(2 / 3, interpolation="linear").alias("p67"),
        ])
    )

    long_pf = 3 if direction == 1 else 1
    short_pf = 1 if direction == 1 else 3

    return (
        df.filter(pl.col(char_col).is_not_null())
        .join(bps, on="eom", how="left")
        .with_columns(
            pl.when(pl.col(char_col) <= pl.col("p33"))
            .then(pl.lit(1))
            .when(pl.col(char_col) <= pl.col("p67"))
            .then(pl.lit(2))
            .otherwise(pl.lit(3))
            .alias("pf")
        )
        .drop(["p33", "p67"])
        .pipe(_value_weight, ["eom", "pf"])
        .with_columns(
            pl.when(pl.col("pf") == long_pf).then(pl.col("vw"))
            .when(pl.col("pf") == short_pf).then(-pl.col("vw"))
            .otherwise(0.0)
            .alias(weight_name)
        )
        .filter(pl.col(weight_name) != 0.0)
        .select(["eom", "id", weight_name])
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_thesis_factors(*, output_dir: Path) -> None:
    """
    Compute stock-level factor weights for all 9 thesis factors and write output.

    Steps:
        1) Load filtered USA characteristics from processed/characteristics/USA.parquet.
        2) Compute Group C weights (MktRF, HML, SMB) from be_me sort.
        3) Compute Group B weights (RMW, CMA, MOM) from FF 2×3 sorts.
        4) Compute Group A weights: BAB (Frazzini-Pedersen), ROE and IA (q-factor 2×3).
        5) Join all factor weights on (eom, id) and write parquet.

    Output:
        processed/thesis_factor_weights.parquet
    """
    data_path = output_dir / "processed"
    char_path = data_path / "characteristics" / "USA.parquet"

    required_cols = [
        "id", "eom", "me", "be_me", "ope_be", "at_gr1", "niq_be",
        "betabab_1260d", "ret_12_1", "ret_exc",
        "crsp_exchcd", "comp_exchg", "size_grp",
        "primary_sec", "common", "obs_main", "exch_main",
    ]

    available = pl.scan_parquet(str(char_path)).columns
    select_cols = [c for c in required_cols if c in available]
    missing = set(required_cols) - set(available)
    if missing:
        print(f"Warning: columns not found in USA.parquet, will be skipped: {missing}")

    print("Loading USA characteristics...")
    df = (
        pl.scan_parquet(str(char_path))
        .select(select_cols)
        .collect()
        .filter(_MAIN_FILTER)
        .sort(["id", "eom"])
    )
    print(f"  {len(df):,} stock-months after filtering")

    # --- Group C ---
    print("Computing MktRF...")
    mktrf = compute_mktrf(df)

    print("Computing HML + SMB...")
    hml, smb = compute_hml_smb(df)

    # --- Group B ---
    print("Computing RMW...")
    rmw = compute_rmw(df)

    print("Computing CMA...")
    cma = compute_cma(df)

    print("Computing MOM...")
    mom = compute_mom(df)

    # --- Group A ---
    print("Computing BAB...")
    bab = compute_bab(df)

    print("Computing ROE...")
    roe = compute_roe(df)

    print("Computing IA...")
    ia = compute_ia(df)

    # --- Join all factors ---
    print("Joining factor weights...")
    universe = df.select(["eom", "id"])

    result = (
        universe
        .join(mktrf, on=["eom", "id"], how="left")
        .join(smb, on=["eom", "id"], how="left")
        .join(hml, on=["eom", "id"], how="left")
        .join(mom, on=["eom", "id"], how="left")
        .join(rmw, on=["eom", "id"], how="left")
        .join(cma, on=["eom", "id"], how="left")
        .join(roe, on=["eom", "id"], how="left")
        .join(ia, on=["eom", "id"], how="left")
        .join(bab, on=["eom", "id"], how="left")
        .fill_null(0.0)
    )

    # Drop rows where all factor weights are zero
    result = result.filter(
        pl.any_horizontal([pl.col(c) != 0.0 for c in _WEIGHT_COLS])
    )

    out_path = data_path / "thesis_factor_weights.parquet"
    result.write_parquet(str(out_path))
    print(f"Wrote {len(result):,} rows to {out_path}")
