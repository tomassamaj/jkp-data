import os
import time
import warnings
from pathlib import Path

import polars as pl

from .config import (
    COLLECT_CHUNK_SIZE,
    END_DATE,
    PORTFOLIO_BP_MIN_N,
    PORTFOLIO_PFS,
    REGIONAL_COUNTRIES_MIN,
    REGIONAL_COUNTRY_EXCL,
    REGIONAL_MONTHS_MIN,
    REGIONAL_STOCKS_MIN,
)
from .output_writer import (
    configure_output_format,
    convert_outputs_to_csv,
    write_dataframe,
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Sortedness.*by.*provided",
)


# =============================================================================
# Helper Functions
# =============================================================================


def add_ecdf(
    df: pl.DataFrame | pl.LazyFrame,
    group_cols: list[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Attach an empirical-CDF ``cdf`` column per group.

    Description:
        Builds the ECDF of the ``var`` column over rows where ``bp_stock``
        is true (the breakpoint sample), then asof-joins the CDF values back
        onto every row of ``df`` — bp and non-bp alike — within each group
        defined by ``group_cols``.

    Steps:
        1) Count ``var`` occurrences per distinct value within each group on
           the bp-stock sub-frame.
        2) Build ``cdf_val`` as a cumulative share within each group.
        3) Asof-join ``(var)`` within ``group_cols`` onto the full input
           frame; non-bp rows pick up the cdf of the nearest bp value ≤ their
           own ``var``, and rows below any bp value fall back to ``0.0``.

    Output:
        Returns the same container type as the input — LazyFrame in, LazyFrame
        out; DataFrame in, DataFrame out — with a ``cdf`` column appended and
        the original columns preserved.
    """
    if group_cols is None:
        group_cols = ["eom"]
    # 1) counts of reference sample per distinct var within each group
    ref_counts = df.filter(pl.col("bp_stock")).group_by(group_cols + ["var"]).agg(n_ref=pl.len())

    # 2) ECDF steps: cumulative share within each group
    ref_steps = (
        ref_counts.sort(group_cols + ["var"])
        .with_columns(
            # apply the window to the whole fraction to ensure same partition
            cdf_val=(pl.cum_sum("n_ref") / pl.sum("n_ref")).over(group_cols)
        )
        .select(group_cols + ["var", "cdf_val"])
    )

    # 3) MUST pre-sort both sides by group_cols + ["var"] for join_asof with 'by'
    left = df.sort(group_cols + ["var"])
    right = ref_steps.sort(group_cols + ["var"])  # already sorted above

    out = (
        left.join_asof(
            right,
            on="var",
            by=group_cols,
            strategy="backward",
        )
        .with_columns(pl.col("cdf_val").fill_null(0.0).alias("cdf"))
        .drop("cdf_val")
    )
    return out


def _build_industry_daily_returns(
    data: pl.DataFrame,
    daily: pl.DataFrame,
    industry_col: str,
    bp_min_n: int,
    excntry: str,
    industry_transform: pl.Expr | None = None,
) -> pl.DataFrame:
    """Description:
        Build daily industry portfolio returns from monthly formation-month weights.
    Steps:
        1) Filter to rows where industry_col is non-null; select id, eom, industry, me, me_cap.
        2) Optionally apply industry_transform to recode the industry column.
        3) Drop industry-month groups with fewer than bp_min_n stocks.
        4) Compute EW/VW/VW-cap weights within each (eom, industry) group.
        5) Join weights to daily returns for the following month.
        6) Aggregate weighted returns by (industry, date).
    Output:
        DataFrame with columns [industry_col, date, n, ret_ew, ret_vw, ret_vw_cap, excntry].
    """
    weights_data = data.filter(pl.col(industry_col).is_not_null()).select(
        ["id", "eom", industry_col, "me", "me_cap"]
    )
    if industry_transform is not None:
        weights_data = weights_data.with_columns(industry_transform)

    weights_data = (
        weights_data.with_columns(pl.len().over([industry_col, "eom"]).alias("bp_n"))
        .filter(pl.col("bp_n") >= bp_min_n)
        .drop("bp_n")
    )

    weights = (
        weights_data.group_by(["eom", industry_col])
        .agg(
            [
                pl.col("id"),
                (1 / pl.len()).alias("w_ew"),
                (pl.col("me") / pl.col("me").sum()).alias("w_vw"),
                (pl.col("me_cap") / pl.col("me_cap").sum()).alias("w_vw_cap"),
            ]
        )
        .explode("id", "w_vw", "w_vw_cap")
    )

    result = (
        weights.lazy()
        .join(
            daily.lazy(),
            left_on=["id", "eom"],
            right_on=["id", "eom_lag1"],
            how="left",
        )
        .filter(pl.col(industry_col).is_not_null() & pl.col("ret_exc").is_not_null())
        .group_by([industry_col, "date"])
        .agg(
            [
                pl.len().alias("n"),
                (pl.col("w_ew") * pl.col("ret_exc")).sum().alias("ret_ew"),
                (pl.col("w_vw") * pl.col("ret_exc")).sum().alias("ret_vw"),
                (pl.col("w_vw_cap") * pl.col("ret_exc")).sum().alias("ret_vw_cap"),
            ]
        )
        .collect()
    )
    return result.with_columns(pl.lit(excntry).str.to_uppercase().alias("excntry"))


# main portfolios function to create the portfolios
def portfolios(
    data_path,
    excntry,
    chars,
    pfs,  # Number of portfolios
    bps,  # What should breakpoints be based on? Non-Microcap stocks ("non_mc") or NYSE stocks "nyse"
    bp_min_n,  # Minimum number of stocks used for breakpoints
    nyse_size_cutoffs,  # Data frame with NYSE size breakpoints
    source=None,  # Use data from "CRSP", "Compustat" or both: ["CRSP", "COMPUSTAT"]. Default: both.
    wins_ret=True,  # Should Compustat returns be winsorized at the 0.1% and 99.9% of CRSP returns?
    cmp_key=False,  # Create characteristics managed size portfolios?
    signals=False,  # Create portfolio signals?
    signals_standardize=False,  # Map chars to [-0.5, +0.5]?,
    signals_w="vw_cap",  # Weighting for signals: in c("ew", "vw", "vw_cap")
    daily_pf=False,  # Should daily return be estimated
    ind_pf=True,  # Should industry portfolio returns be estimated
    ret_cutoffs=None,  # Data frame for monthly winsorization. Neccesary when wins_ret=T
    ret_cutoffs_daily=None,  # Data frame for daily winsorization. Neccesary when wins_ret=T and daily_pf=T
):
    if source is None:
        source = ["CRSP", "COMPUSTAT"]
    # characerteristics data
    file_path = f"{data_path}/characteristics/{excntry}.parquet"

    # Select the required columns
    columns = (
        [
            "id",
            "eom",
            "source_crsp",
            "comp_exchg",
            "crsp_exchcd",
            "size_grp",
            "ret_exc",
            "ret_exc_lead1m",
            "me",
            "gics",
            "ff49",
        ]
        + chars
        + ["excntry"]
    )

    # Build the full preprocessing chain as a single lazy pipeline. This lets
    # polars push predicate/null filters into the parquet reader (skipping row
    # groups on real production files) and fuse the ~15 intermediate steps into
    # one pass instead of allocating ~15 intermediate DataFrames.
    cast_exclude = {"id", "eom", "source_crsp", "size_grp", "excntry"}
    cast_cols = [c for c in columns if c not in cast_exclude]

    if bps == "nyse":
        bp_stock_expr = (
            ((pl.col("crsp_exchcd") == 1) & pl.col("comp_exchg").is_null())
            | ((pl.col("comp_exchg") == 11) & pl.col("crsp_exchcd").is_null())
        ).alias("bp_stock")
    else:  # "non_mc"
        bp_stock_expr = pl.col("size_grp").is_in(["mega", "large", "small"]).alias("bp_stock")

    data_lazy = (
        pl.scan_parquet(file_path)
        .select(columns)
        .join(nyse_size_cutoffs.lazy().select(["eom", "nyse_p80"]), on="eom", how="left")
        .with_columns(pl.min_horizontal(pl.col("me"), pl.col("nyse_p80")).alias("me_cap"))
        .drop("nyse_p80")
        .with_columns([pl.col(c).cast(pl.Float64) for c in cast_cols])
        .filter(
            (pl.col("size_grp").is_not_null())
            & (pl.col("me").is_not_null())
            & (pl.col("ret_exc_lead1m").is_not_null())
        )
        .with_columns(bp_stock_expr)
    )

    # Conditional source screen
    if len(source) == 1:
        if source[0] == "CRSP":
            data_lazy = data_lazy.filter(pl.col("source_crsp") == 1)
        elif source[0] == "COMPUSTAT":
            data_lazy = data_lazy.filter(pl.col("source_crsp") == 0)

    # Daily returns as a lazy chain (scan_parquet + winsorization).
    daily_file_path = f"{data_path}/return_data/daily_rets_by_country/{excntry}.parquet"
    if daily_pf:
        daily_lazy = (
            pl.scan_parquet(daily_file_path)
            .select(["id", "date", "ret_exc"])
            .with_columns((pl.col("date").dt.month_start().dt.offset_by("-1d")).alias("eom_lag1"))
            .with_columns(pl.col("ret_exc").cast(pl.Float64))
        )
    else:
        daily_lazy = None

    # Monthly winsorization: clip Compustat ret_exc_lead1m to CRSP quantiles.
    if wins_ret:
        data_lazy = (
            data_lazy.join(
                ret_cutoffs.lazy()
                .select(["eom_lag1", "ret_exc_0_1", "ret_exc_99_9"])
                .rename({"eom_lag1": "eom", "ret_exc_0_1": "p001", "ret_exc_99_9": "p999"}),
                on="eom",
                how="left",
            )
            .with_columns(
                pl.when((pl.col("source_crsp") == 0) & (pl.col("ret_exc_lead1m") > pl.col("p999")))
                .then(pl.col("p999"))
                .when((pl.col("source_crsp") == 0) & (pl.col("ret_exc_lead1m") < pl.col("p001")))
                .then(pl.col("p001"))
                .otherwise(pl.col("ret_exc_lead1m"))
                .alias("ret_exc_lead1m")
            )
            .drop(["source_crsp", "p001", "p999"])
        )

        # Daily winsorization
        if daily_pf:
            daily_lazy = (
                daily_lazy.with_columns(pl.col("date").dt.month_end().alias("eom"))
                .join(
                    ret_cutoffs_daily.lazy()
                    .select(["eom", "ret_exc_0_1", "ret_exc_99_9"])
                    .rename({"ret_exc_0_1": "p001", "ret_exc_99_9": "p999"}),
                    on="eom",
                    how="left",
                )
                .with_columns(
                    pl.when((pl.col("id") > 99999) & (pl.col("ret_exc") > pl.col("p999")))
                    .then(pl.col("p999"))
                    .when((pl.col("id") > 99999) & (pl.col("ret_exc") < pl.col("p001")))
                    .then(pl.col("p001"))
                    .otherwise(pl.col("ret_exc"))
                    .alias("ret_exc")
                )
                .drop(["p001", "p999", "eom"])
            )

    # Collect the lazy chain once — the single fused read + preprocess pass.
    data = data_lazy.collect()

    # Collect daily lazy frame if present (needed for industry returns and per-char daily).
    daily = daily_lazy.collect() if daily_lazy is not None else None

    # standardizing signals
    if signals_standardize and signals:
        data = (
            data
            # Ranking within groups defined by 'eom'
            .with_columns(
                [
                    (pl.col(char).rank(method="min").over("eom").cast(pl.Int64)).alias(char)
                    for char in chars
                ]
            )
            # normalizing ranks
            .with_columns(
                [
                    (((pl.col(char) / pl.col(char).max()) - pl.lit(0.5)).over("eom")).alias(char)
                    for char in chars
                ]
            )
        )

    if ind_pf:
        # Filter data where 'gics' is not null and select required columns
        ind_data = data.filter(pl.col("gics").is_not_null()).select(
            ["eom", "gics", "excntry", "ret_exc_lead1m", "me", "me_cap"]
        )

        # Process GICS codes (extract first 2 digits and convert to numeric)
        ind_data = ind_data.with_columns(
            (pl.col("gics").cast(pl.Utf8).str.slice(0, 2).cast(pl.Int64)).alias("gics")
        )

        # Calculate industry returns based on GICS
        ind_gics = ind_data.group_by(["gics", "eom"]).agg(
            [
                pl.len().alias("n"),
                (pl.col("ret_exc_lead1m").mean()).alias("ret_ew"),
                ((pl.col("ret_exc_lead1m") * pl.col("me")).sum() / pl.col("me").sum()).alias(
                    "ret_vw"
                ),
                (
                    (pl.col("ret_exc_lead1m") * pl.col("me_cap")).sum() / pl.col("me_cap").sum()
                ).alias("ret_vw_cap"),
            ]
        )
        ind_gics = ind_gics.with_columns(pl.lit(excntry).str.to_uppercase().alias("excntry"))
        ind_gics = ind_gics.with_columns(
            (pl.col("eom").dt.offset_by("1mo").dt.month_end()).alias("eom")
        )
        ind_gics = ind_gics.filter(pl.col("n") >= bp_min_n)

        # Estimate industry portfolios by Fama-French portfolios for US data
        if excntry.lower() == "usa":
            ind_data = data.filter(pl.col("ff49").is_not_null()).select(
                ["eom", "ff49", "ret_exc_lead1m", "me", "me_cap"]
            )
            ind_ff49 = ind_data.group_by(["ff49", "eom"]).agg(
                [
                    pl.len().alias("n"),
                    (pl.col("ret_exc_lead1m").mean()).alias("ret_ew"),
                    ((pl.col("ret_exc_lead1m") * pl.col("me")).sum() / pl.col("me").sum()).alias(
                        "ret_vw"
                    ),
                    (
                        (pl.col("ret_exc_lead1m") * pl.col("me_cap")).sum() / pl.col("me_cap").sum()
                    ).alias("ret_vw_cap"),
                ]
            )
            ind_ff49 = ind_ff49.with_columns(pl.lit(excntry).str.to_uppercase().alias("excntry"))
            ind_ff49 = ind_ff49.with_columns(
                (pl.col("eom").dt.offset_by("1mo").dt.month_end()).alias("eom")
            )
            ind_ff49 = ind_ff49.filter(pl.col("n") >= bp_min_n)

        if daily_pf:
            ind_gics_daily = _build_industry_daily_returns(
                data,
                daily,
                "gics",
                bp_min_n,
                excntry,
                industry_transform=pl.col("gics")
                .cast(pl.Utf8)
                .str.slice(0, 2)
                .cast(pl.Int64)
                .alias("gics"),
            )

            if excntry.lower() == "usa":
                ind_ff49_daily = _build_industry_daily_returns(
                    data,
                    daily,
                    "ff49",
                    bp_min_n,
                    excntry,
                )

    # Creating portfolios for all the characteristics.
    #
    # Each characteristic builds a per-char lazy pipeline (ECDF -> pf
    # assignment -> monthly/daily aggregation).  The pipelines accumulate in
    # ``pf_returns_lazys`` / ``pf_daily_lazys`` and are batch-collected via
    # ``pl.collect_all`` so polars can execute them concurrently on its thread
    # pool.  The signals=True branch is the exception: it needs the
    # intermediate ``sub`` frame alive after each collect to compute
    # pf_signals, so it collects eagerly per-char.
    pf_returns_lazys: list[pl.LazyFrame] = []
    pf_daily_lazys: list[pl.LazyFrame] = []
    char_pfs: list[dict] = []  # populated only when signals=True

    # Single shared LazyFrame node for the preprocessed data. All per-char
    # pipelines branch off this node, enabling polars' common-subplan
    # elimination (CSE) in collect_all to avoid redundant scans.
    data_lazy = data.lazy()

    for x in chars:
        # Alias current char into a 'var' column on the per-char subset.
        # Operate on `sub` only -- `data` is not mutated.
        if not signals:
            sub = (
                data_lazy.with_columns(pl.col(x).cast(pl.Float64).alias("var"))
                .filter(pl.col("var").is_not_null())
                .select(
                    [
                        "id",
                        "eom",
                        "var",
                        "size_grp",
                        "ret_exc_lead1m",
                        "me",
                        "me_cap",
                        "crsp_exchcd",
                        "comp_exchg",
                        "bp_stock",
                    ]
                )
            )
        else:
            sub = data_lazy.with_columns(pl.col(x).cast(pl.Float64).alias("var")).filter(
                pl.col("var").is_not_null()
            )

        sub = sub.with_columns(bp_n=pl.sum("bp_stock").over("eom")).filter(
            pl.col("bp_n") >= bp_min_n
        )

        # Skip chars with no data after the bp_n filter.  For the
        # collect_all path we cannot cheaply check emptiness without an
        # eager collect, so we optimistically build the lazy pipeline and
        # drop empty results after the batch collect.  For signals=True
        # we still need the eager gate because the pf_signals block
        # requires intermediate access to ``sub``.
        if signals and sub.limit(1).collect().height == 0:
            continue

        sub = add_ecdf(sub)
        sub = sub.with_columns(pl.col("cdf").min().over("eom").alias("min_cdf"))
        sub = sub.with_columns(
            pl.when(pl.col("cdf") == pl.col("min_cdf"))
            .then(0.00000001)
            .otherwise(pl.col("cdf"))
            .alias("cdf")
        )
        sub = sub.with_columns(
            (pl.col("cdf") * pfs).ceil().clip(lower_bound=1, upper_bound=pfs).alias("pf")
        )

        # Monthly pf_returns lazy frame for this char.
        pf_returns_x = (
            sub.group_by(["pf", "eom"])
            .agg(
                [
                    pl.lit(x).alias("characteristic"),
                    pl.len().alias("n"),
                    pl.median("var").alias("signal"),
                    pl.mean("ret_exc_lead1m").alias("ret_ew"),
                    ((pl.col("ret_exc_lead1m") * pl.col("me")).sum() / pl.col("me").sum()).alias(
                        "ret_vw"
                    ),
                    (
                        (pl.col("ret_exc_lead1m") * pl.col("me_cap")).sum() / pl.col("me_cap").sum()
                    ).alias("ret_vw_cap"),
                ]
            )
            .with_columns(pl.col("eom").dt.offset_by("1mo").dt.month_end().alias("eom"))
        )

        if signals:
            # Eager path: collect immediately, then compute pf_signals.
            op: dict = {}
            op["pf_returns"] = pf_returns_x.collect()

            if signals_w == "ew":
                sub = sub.with_columns((1 / pl.col("eom").len()).over(["pf", "eom"]).alias("w"))
            elif signals_w == "vw":
                sub = sub.with_columns(
                    (pl.col("me") / pl.col("me").sum()).over(["pf", "eom"]).alias("w")
                )
            elif signals_w == "vw_cap":
                sub = sub.with_columns(
                    (pl.col("me_cap") / pl.col("me_cap").sum()).over(["pf", "eom"]).alias("w")
                )

            sub = sub.with_columns(
                [
                    pl.when(pl.col(var).is_null()).then(pl.lit(0)).otherwise(pl.col(var)).alias(var)
                    for var in chars
                ]
            )
            pf_signals = sub.with_columns(
                [(pl.col("w") * pl.col(var)).sum().over(["pf", "eom"]) for var in chars]
            )
            pf_signals = pf_signals.with_columns(
                [
                    pl.lit(x).alias("characteristic"),
                    pl.col("eom").dt.offset_by("1mo").dt.month_end().alias("eom"),
                ]
            )
            op["signals"] = pf_signals.collect()

            if daily_pf:
                weights = (
                    sub.group_by(["eom", "pf"])
                    .agg(
                        [
                            pl.col("id"),
                            (1 / pl.len()).alias("w_ew"),
                            (pl.col("me") / pl.col("me").sum()).alias("w_vw"),
                            (pl.col("me_cap") / pl.col("me_cap").sum()).alias("w_vw_cap"),
                        ]
                    )
                    .explode("id", "w_vw", "w_vw_cap")
                )
                daily_sub = weights.join(
                    daily.lazy(),
                    left_on=["id", "eom"],
                    right_on=["id", "eom_lag1"],
                    how="left",
                ).filter((pl.col("pf").is_not_null()) & (pl.col("ret_exc").is_not_null()))
                pf_daily_x = daily_sub.group_by(["pf", "date"]).agg(
                    [
                        pl.lit(x).alias("characteristic"),
                        pl.len().alias("n"),
                        ((pl.col("w_ew") * pl.col("ret_exc")).sum()).alias("ret_ew"),
                        ((pl.col("w_vw") * pl.col("ret_exc")).sum()).alias("ret_vw"),
                        ((pl.col("w_vw_cap") * pl.col("ret_exc")).sum()).alias("ret_vw_cap"),
                    ]
                )
                op["pf_daily"] = pf_daily_x.collect()

            char_pfs.append(op)
        else:
            # Lazy path: accumulate LazyFrames for batch collect_all.
            pf_returns_lazys.append(pf_returns_x)

            if daily_pf:
                weights = (
                    sub.group_by(["eom", "pf"])
                    .agg(
                        [
                            pl.col("id"),
                            (1 / pl.len()).alias("w_ew"),
                            (pl.col("me") / pl.col("me").sum()).alias("w_vw"),
                            (pl.col("me_cap") / pl.col("me_cap").sum()).alias("w_vw_cap"),
                        ]
                    )
                    .explode("id", "w_vw", "w_vw_cap")
                )
                daily_sub = weights.join(
                    daily.lazy(),
                    left_on=["id", "eom"],
                    right_on=["id", "eom_lag1"],
                    how="left",
                ).filter((pl.col("pf").is_not_null()) & (pl.col("ret_exc").is_not_null()))
                pf_daily_x = daily_sub.group_by(["pf", "date"]).agg(
                    [
                        pl.lit(x).alias("characteristic"),
                        pl.len().alias("n"),
                        ((pl.col("w_ew") * pl.col("ret_exc")).sum()).alias("ret_ew"),
                        ((pl.col("w_vw") * pl.col("ret_exc")).sum()).alias("ret_vw"),
                        ((pl.col("w_vw_cap") * pl.col("ret_exc")).sum()).alias("ret_vw_cap"),
                    ]
                )
                pf_daily_lazys.append(pf_daily_x)

    # Batch-collect per-char lazy pipelines in chunks to bound peak memory
    # (see COLLECT_CHUNK_SIZE in config.py).
    pf_returns_df: pl.DataFrame | None = None
    pf_daily_df: pl.DataFrame | None = None

    if signals and char_pfs:
        pf_returns_df = pl.concat([op["pf_returns"] for op in char_pfs])
        if daily_pf:
            pf_daily_df = pl.concat([op["pf_daily"] for op in char_pfs])
    elif pf_returns_lazys:
        ret_dfs: list[pl.DataFrame] = []
        daily_dfs: list[pl.DataFrame] = []
        n_chars = len(pf_returns_lazys)
        n_chunks = (n_chars + COLLECT_CHUNK_SIZE - 1) // COLLECT_CHUNK_SIZE
        for chunk_idx, start in enumerate(range(0, n_chars, COLLECT_CHUNK_SIZE)):
            end = min(start + COLLECT_CHUNK_SIZE, n_chars)
            print(
                f"   Chars {start + 1}-{end} of {n_chars} (chunk {chunk_idx + 1}/{n_chunks})",
                flush=True,
            )
            chunk_ret = pf_returns_lazys[start:end]
            chunk_daily = pf_daily_lazys[start:end] if daily_pf else []
            collected = pl.collect_all(chunk_ret + chunk_daily)
            n_ret_chunk = len(chunk_ret)
            ret_dfs.extend(df for df in collected[:n_ret_chunk] if df.height > 0)
            if daily_pf:
                daily_dfs.extend(df for df in collected[n_ret_chunk:] if df.height > 0)
        pf_returns_df = pl.concat(ret_dfs) if ret_dfs else None
        if daily_pf:
            pf_daily_df = pl.concat(daily_dfs) if daily_dfs else None

    output = {}
    if pf_returns_df is not None:
        output["pf_returns"] = pf_returns_df
    if daily_pf and pf_daily_df is not None:
        output["pf_daily"] = pf_daily_df
    # Handle industry portfolio returns if ind_pf is true
    if ind_pf:
        output["gics_returns"] = ind_gics
        if excntry.lower() == "usa":
            output["ff49_returns"] = ind_ff49
        if daily_pf:
            output["gics_daily"] = ind_gics_daily
            if excntry.lower() == "usa":
                output["ff49_daily"] = ind_ff49_daily

    # Add excntry to pf_returns and pf_daily, and aggregate signals
    if len(output) > 0:
        if "pf_returns" in output and output["pf_returns"].height > 0:
            output["pf_returns"] = output["pf_returns"].with_columns(
                pl.lit(excntry).str.to_uppercase().alias("excntry")
            )
            if daily_pf and "pf_daily" in output:
                output["pf_daily"] = output["pf_daily"].with_columns(
                    pl.lit(excntry).str.to_uppercase().alias("excntry")
                )
            if signals and "signals" in output:
                output["signals"] = pl.concat([op["signals"] for op in char_pfs])
                output["signals"] = output["signals"].with_columns(
                    pl.lit(excntry).str.to_uppercase().alias("excntry")
                )

    results = []
    if cmp_key:
        for x in chars:
            print(f"   CMP - {x}: {chars.index(x) + 1} out of {len(chars)}")

            # Create a new column 'var' based on the current 'x'
            data = data.with_columns(pl.col(x).alias("var"))

            # Subsetting and ranking
            sub = data.filter(pl.col("var").is_not_null()).select(
                ["eom", "var", "size_grp", "ret_exc_lead1m"]
            )

            # Calculate ranks, rank deviations, and weights
            sub = (
                sub.with_columns(
                    (
                        (pl.col("var").rank("average").over("size_grp", "eom"))
                        / (pl.len().over("size_grp", "eom") + 1)
                    ).alias("p_rank")
                )
                .with_columns(pl.col("p_rank").mean().over("size_grp", "eom").alias("mean_p_rank"))
                .with_columns((pl.col("p_rank") - pl.col("mean_p_rank")).alias("p_rank_dev"))
                .with_columns(
                    (pl.col("p_rank_dev") / ((pl.col("p_rank_dev").abs().sum()) / 2))
                    .over("size_grp", "eom")
                    .alias("weight")
                )
            )

            # Aggregation
            cmp = (
                sub.group_by(["size_grp", "eom"])
                .agg(
                    [
                        pl.lit(x).alias("characteristic"),
                        pl.len().alias("n_stocks"),
                        ((pl.col("ret_exc_lead1m") * pl.col("weight")).sum()).alias("ret_weighted"),
                        ((pl.col("var") * pl.col("weight")).sum()).alias("signal_weighted"),
                        pl.col("var").std().alias("sd_var"),
                    ]
                )
                .with_columns(pl.lit(excntry).alias("excntry"))
            )

            # Post-processing
            cmp = cmp.filter(pl.col("sd_var") != 0).drop("sd_var")
            cmp = cmp.with_columns((pl.col("eom").dt.offset_by("1mo").dt.month_end()).alias("eom"))

            results.append(cmp)

    if len(results) > 0:
        output_cmp = pl.concat(results)
        output_cmp = output_cmp.with_columns(pl.col("excntry").str.to_uppercase().alias("excntry"))
        output["cmp"] = output_cmp

    return output


# function for regional grouping of portfolios etc
def regional_data(
    data,
    mkt,
    date_col,
    char_col,
    countries,
    weighting,
    countries_min,
    periods_min,
    stocks_min,
):
    # Determine Country Weights
    weights = mkt.select(
        [
            pl.col("excntry"),
            pl.col(date_col).alias(date_col),
            pl.col("mkt_vw_exc"),
            pl.when(weighting == "market_cap")
            .then(pl.col("me_lag1"))
            .when(weighting == "stocks")
            .then(pl.col("stocks").cast(pl.Float64))
            .when(weighting == "ew")
            .then(1)
            .alias("country_weight"),
        ]
    )
    # Portfolio Return
    pf = data.filter(
        (pl.col("excntry").is_in(countries.implode())) & (pl.col("n_stocks_min") >= stocks_min)
    )
    pf = pf.join(weights, on=["excntry", date_col], how="left")
    pf = (
        pf.filter(pl.col("mkt_vw_exc").is_not_null())
        .group_by([char_col, date_col])
        .agg(
            [
                pl.len().alias("n_countries"),
                pl.col("direction").first().alias("direction"),
                (pl.col("ret_ew") * pl.col("country_weight")).sum()
                / pl.col("country_weight").sum().alias("ret_ew"),
                (pl.col("ret_vw") * pl.col("country_weight")).sum()
                / pl.col("country_weight").sum().alias("ret_vw"),
                (pl.col("ret_vw_cap") * pl.col("country_weight")).sum()
                / pl.col("country_weight").sum().alias("ret_vw_cap"),
                (pl.col("mkt_vw_exc") * pl.col("country_weight")).sum()
                / pl.col("country_weight").sum().alias("mkt_vw_exc"),
            ]
        )
    )

    # Minimum Requirement: Countries
    pf = pf.filter(pl.col("n_countries") >= countries_min)

    # Minimum Requirement: Months
    pf = (
        pf.with_columns(pl.len().over(char_col).alias("periods"))
        .filter(pl.col("periods") >= periods_min)
        .drop("periods")
        .sort([char_col, date_col])
    )

    return pf


# =============================================================================
# Main Entry Point
# =============================================================================


def run_portfolio(*, output_format: str = "parquet", output_dir: Path) -> None:
    """Run portfolio generation for Group A thesis factors (USA only).

    Description:
        Build factor portfolios for betabab_1260d, niq_be, and at_gr1 for US
        stocks only. Produces monthly long-short factor returns and portfolio
        assignments that thesis_factors.py uses to read stock-level weights.
    Steps:
        1) Load characteristic definitions and breakpoint data.
        2) Construct monthly portfolios for USA.
        3) Aggregate long-minus-short factor returns.
        4) Write pfs.parquet, hml.parquet, lms.parquet, and country_factors/USA.parquet.
    Output:
        Portfolio files written to data/processed/portfolios/.
    """
    configure_output_format(output_format)

    # Setting data path and output path
    data_path = str(output_dir / "processed")
    output_path = str(output_dir / "processed" / "portfolios")

    # Get list of countries from characteristics files
    # Only USA is needed for the thesis
    countries = ["USA"]

    # Only the 3 Group-A characteristics needed for thesis factors
    chars = [
        "betabab_1260d",
        "niq_be",
        "at_gr1",
    ]

    settings = {
        "end_date": END_DATE,
        "pfs": PORTFOLIO_PFS,
        "source": ["CRSP", "COMPUSTAT"],
        "wins_ret": True,
        "bps": "non_mc",
        "bp_min_n": PORTFOLIO_BP_MIN_N,
    }

    print(
        f"Start          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
        flush=True,
    )

    from .paths import get_factor_details_path

    char_info = (
        pl.read_excel(
            get_factor_details_path(),
            sheet_name="details",
        )
        .filter(pl.col("abr_jkp").is_not_null())
        .select([pl.col("abr_jkp").alias("characteristic"), pl.col("direction").cast(pl.Int32)])
    )

    nyse_size_cutoffs = pl.read_parquet(f"{data_path}/other_output/nyse_cutoffs.parquet")

    ret_cutoffs = pl.read_parquet(f"{data_path}/other_output/return_cutoffs.parquet")
    ret_cutoffs = ret_cutoffs.with_columns(
        (pl.col("eom").dt.month_start().dt.offset_by("-1d")).alias("eom_lag1")
    )

    # Build monthly portfolios for USA only
    portfolio_data = {}
    for ex in countries:
        print(f"{ex}: {countries.index(ex) + 1} out of {len(countries)}")
        result = portfolios(
            data_path=data_path,
            excntry=ex,
            chars=chars,
            pfs=settings["pfs"],
            bps=settings["bps"],
            bp_min_n=settings["bp_min_n"],
            nyse_size_cutoffs=nyse_size_cutoffs,
            source=settings["source"],
            wins_ret=settings["wins_ret"],
            cmp_key=False,
            signals=False,
            signals_standardize=True,
            signals_w="vw_cap",
            daily_pf=False,
            ind_pf=False,
            ret_cutoffs=ret_cutoffs,
            ret_cutoffs_daily=None,
        )
        portfolio_data[ex] = result

    # Aggregate portfolio returns across countries (USA only)
    if any(sub_data and "pf_returns" in sub_data for sub_key, sub_data in portfolio_data.items()):
        pf_returns = pl.concat(
            [
                sub_data["pf_returns"]
                for sub_key, sub_data in portfolio_data.items()
                if sub_data and "pf_returns" in sub_data
            ]
        )
        pf_returns = pf_returns.select(
            [
                "excntry",
                "characteristic",
                "pf",
                "eom",
                "n",
                "signal",
                "ret_ew",
                "ret_vw",
                "ret_vw_cap",
            ]
        )
        pf_returns = pf_returns.sort(["excntry", "characteristic", "pf", "eom"])
    else:
        pf_returns = None

    # Create HML Returns (high-minus-low, pf3 minus pf1)
    if pf_returns is not None and pf_returns.height > 0:
        hml_returns = pf_returns.group_by(["excntry", "characteristic", "eom"]).agg(
            [
                pl.col("pf").is_in([settings["pfs"], 1]).sum().alias("pfs"),
                (
                    pl.col("signal").filter(pl.col("pf") == settings["pfs"]).first()
                    - pl.col("signal").filter(pl.col("pf") == 1).first()
                ).alias("signal"),
                (
                    pl.col("n").filter(pl.col("pf") == settings["pfs"]).first()
                    + pl.col("n").filter(pl.col("pf") == 1).first()
                ).alias("n_stocks"),
                (pl.col("n").filter(pl.col("pf").is_in([settings["pfs"], 1])).min()).alias(
                    "n_stocks_min"
                ),
                (
                    pl.col("ret_ew").filter(pl.col("pf") == settings["pfs"]).first()
                    - pl.col("ret_ew").filter(pl.col("pf") == 1).first()
                ).alias("ret_ew"),
                (
                    pl.col("ret_vw").filter(pl.col("pf") == settings["pfs"]).first()
                    - pl.col("ret_vw").filter(pl.col("pf") == 1).first()
                ).alias("ret_vw"),
                (
                    pl.col("ret_vw_cap").filter(pl.col("pf") == settings["pfs"]).first()
                    - pl.col("ret_vw_cap").filter(pl.col("pf") == 1).first()
                ).alias("ret_vw_cap"),
            ]
        )
        hml_returns = hml_returns.filter(pl.col("pfs") == 2).drop("pfs")
        hml_returns = hml_returns.sort(["excntry", "characteristic", "eom"])

        # Sign returns consistent with factor direction (direction=-1 means short high)
        lms_returns = char_info.join(hml_returns, on="characteristic", how="left")
        resign_cols = ["signal", "ret_ew", "ret_vw", "ret_vw_cap"]
        lms_returns = lms_returns.with_columns(
            [pl.col(var) * pl.col("direction").alias(var) for var in resign_cols]
        )
    else:
        hml_returns = None
        lms_returns = None

    # Writing output
    os.makedirs(output_path, exist_ok=True)
    if pf_returns is not None:
        write_dataframe(
            pf_returns.filter(pl.col("eom") <= settings["end_date"]),
            f"{output_path}/pfs.parquet",
        )
    if hml_returns is not None:
        write_dataframe(
            hml_returns.filter(pl.col("eom") <= settings["end_date"]),
            f"{output_path}/hml.parquet",
        )
    if lms_returns is not None:
        write_dataframe(
            lms_returns.filter(pl.col("eom") <= settings["end_date"]),
            f"{output_path}/lms.parquet",
        )

    # Write country-level long-short returns for USA
    if lms_returns is not None:
        cnt_folder = os.path.join(output_path, "country_factors")
        os.makedirs(cnt_folder, exist_ok=True)
        for exc in lms_returns["excntry"].unique():
            if exc:
                filtered_df = lms_returns.filter(
                    (pl.col("eom") <= settings["end_date"]) & (pl.col("excntry") == exc)
                )
                write_dataframe(filtered_df, os.path.join(cnt_folder, f"{exc}.parquet"))

    convert_outputs_to_csv(processed_dir=data_path)

    print(
        f"End            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}",
        flush=True,
    )
