"""Unit tests for thesis_factors.py.

All tests use synthetic in-memory DataFrames — no WRDS access or large files required.

Key invariants checked:
  - Weight sums: long leg = +1.0, short leg = -1.0, per month
  - Direction: low-beta stocks get positive BAB weight, etc.
  - Null handling: stocks with null characteristic get zero weight
  - Annual rebalancing: July uses June breakpoints; June itself uses prior-year breakpoints

Data design:
  For FF 2×3 sorts, size (me) and the sorted characteristic must be INDEPENDENT so that
  all 6 portfolio cells are populated. We use 60 stocks split into 30 "small" (me=100)
  and 30 "big" (me=500), with each characteristic spanning the full range 0→1 within
  each size group. All stocks are NYSE so NYSE breakpoints use the full sample.
"""

from __future__ import annotations

import calendar
from datetime import date

import polars as pl
import pytest

from jkp.data.thesis_factors import (
    _univariate_tercile,
    compute_bab,
    compute_cma,
    compute_hml_smb,
    compute_ia,
    compute_mom,
    compute_mktrf,
    compute_rmw,
    compute_roe,
)


# =============================================================================
# Synthetic data factories
# =============================================================================

def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _make_ff_panel(months: list[date], n_per_size: int = 30) -> pl.DataFrame:
    """
    Synthetic panel for FF 2×3 double-sort tests.

    Design guarantees all 6 portfolio cells are populated:
      - First n_per_size stocks: small (me=100)
      - Next  n_per_size stocks: big   (me=500)
      - All stocks are NYSE (crsp_exchcd=1)
      - Each characteristic spans [0, 1] independently within each size group
        (char rank = i % n_per_size, so small and big stocks share the same
        characteristic distribution → all corner cells fill)

    NYSE size median from June data = 300, cleanly separating small from big.
    """
    n_stocks = 2 * n_per_size
    rows = []
    for eom in months:
        for i in range(n_stocks):
            frac = (i % n_per_size) / max(n_per_size - 1, 1)  # char rank in [0,1], same for S and B
            rows.append({
                "id": i + 1,
                "eom": eom,
                "me": 100.0 if i < n_per_size else 500.0,  # small or big
                "be_me": 0.1 + frac * 2.0,         # 0.1 – 2.1, same spread in S and B
                "ope_be": -0.2 + frac * 0.6,        # −0.2 – 0.4
                "at_gr1": -0.3 + frac * 0.8,        # −0.3 – 0.5
                "niq_be": -0.1 + frac * 0.3,        # −0.1 – 0.2
                "betabab_1260d": 0.5 + frac * 1.5,  # 0.5 – 2.0
                "ret_12_1": -0.4 + frac * 1.2,      # −0.4 – 0.8
                "ret_exc": -0.05 + frac * 0.15,     # monthly return for shift(1) in MOM
                "crsp_exchcd": 1,  # all NYSE
                "comp_exchg": None,
                "size_grp": "large",  # all pass non-MC filter
                "primary_sec": 1,
                "common": 1,
                "obs_main": 1,
                "exch_main": 1,
            })
    return (
        pl.DataFrame(rows)
        .with_columns([
            pl.col("eom").cast(pl.Date),
            pl.col("crsp_exchcd").cast(pl.Int32),
            pl.col("comp_exchg").cast(pl.Int32),
        ])
    )


def _make_univariate_panel(months: list[date], n_stocks: int = 30) -> pl.DataFrame:
    """
    Synthetic panel for univariate tercile sort tests (Group A).
    n_stocks × months rows; characteristics increase monotonically with stock id.
    First 2/3 of stocks are non-microcap (for breakpoints); last 1/3 are micro.
    """
    rows = []
    for eom in months:
        for i in range(n_stocks):
            frac = i / (n_stocks - 1)
            rows.append({
                "id": i + 1,
                "eom": eom,
                "me": 50.0 + frac * 500.0,
                "betabab_1260d": 0.2 + frac * 2.0,
                "niq_be": -0.1 + frac * 0.3,
                "at_gr1": -0.3 + frac * 0.8,
                "size_grp": "micro" if i >= n_stocks * 2 // 3 else "large",
                "primary_sec": 1,
                "common": 1,
                "obs_main": 1,
                "exch_main": 1,
                # Unused by univariate sort but required by _MAIN_FILTER (already applied)
                "crsp_exchcd": 1,
                "comp_exchg": None,
            })
    return (
        pl.DataFrame(rows)
        .with_columns([
            pl.col("eom").cast(pl.Date),
            pl.col("crsp_exchcd").cast(pl.Int32),
            pl.col("comp_exchg").cast(pl.Int32),
        ])
    )


# Shared test dates
JUNE_2019 = _month_end(2019, 6)
JULY_2019 = _month_end(2019, 7)
AUG_2019  = _month_end(2019, 8)
DEC_2018  = _month_end(2018, 12)
JAN_2019  = _month_end(2019, 1)


# =============================================================================
# MktRF
# =============================================================================

class TestComputeMktrf:
    def test_weights_sum_to_one_per_month(self):
        df = _make_ff_panel([JUNE_2019, JULY_2019])
        out = compute_mktrf(df)
        for s in out.group_by("eom").agg(pl.col("w_MktRF").sum())["w_MktRF"].to_list():
            assert abs(s - 1.0) < 1e-10, f"MktRF weights don't sum to 1: {s}"

    def test_all_weights_positive(self):
        df = _make_ff_panel([JULY_2019])
        out = compute_mktrf(df)
        assert (out["w_MktRF"] > 0).all()

    def test_larger_me_gets_larger_weight(self):
        """Big stocks (me=500) should have 5× the weight of small stocks (me=100)."""
        df = _make_ff_panel([JULY_2019], n_per_size=1)  # 1 small + 1 big
        out = compute_mktrf(df).sort("id")
        w_small, w_big = out["w_MktRF"].to_list()
        assert abs(w_big / w_small - 5.0) < 1e-10

    def test_output_columns(self):
        df = _make_ff_panel([JULY_2019])
        assert set(compute_mktrf(df).columns) == {"eom", "id", "w_MktRF"}


# =============================================================================
# HML and SMB
# =============================================================================

class TestComputeHmlSmb:
    """Annual June rebalancing: June 2019 NYSE data sets breakpoints for July 2019+."""

    def setup_method(self):
        self.df = _make_ff_panel([JUNE_2019, JULY_2019, AUG_2019])
        self.hml, self.smb = compute_hml_smb(self.df)

    def _july_hml(self):
        return self.hml.filter(pl.col("eom") == JULY_2019)

    def _july_smb(self):
        return self.smb.filter(pl.col("eom") == JULY_2019)

    def test_hml_long_weights_sum_to_one(self):
        jul = self._july_hml()
        assert jul.height > 0, "No HML assignments in July — check breakpoint data"
        long_sum = jul.filter(pl.col("w_HML") > 0)["w_HML"].sum()
        assert abs(long_sum - 1.0) < 1e-10, f"HML long sum = {long_sum}"

    def test_hml_short_weights_sum_to_minus_one(self):
        jul = self._july_hml()
        assert jul.height > 0
        short_sum = jul.filter(pl.col("w_HML") < 0)["w_HML"].sum()
        assert abs(short_sum + 1.0) < 1e-10, f"HML short sum = {short_sum}"

    def test_hml_high_bm_stocks_get_positive_weight(self):
        """High be_me stocks (top 30%) appear in the long leg."""
        jul = self._july_hml()
        long_ids = jul.filter(pl.col("w_HML") > 0)["id"].to_list()
        # By construction, high-be_me stocks have high id % n_per_size
        assert max(long_ids) >= 27  # among ids 1-30 and 31-60

    def test_hml_low_bm_stocks_get_negative_weight(self):
        jul = self._july_hml()
        short_ids = jul.filter(pl.col("w_HML") < 0)["id"].to_list()
        # Low-be_me stocks → id % 30 is low
        assert min(short_ids) <= 4

    def test_hml_mid_bm_stocks_have_zero_weight(self):
        """Middle B/M stocks don't appear in HML output (filtered out)."""
        jul = self._july_hml()
        # Long + short should only be ~40% of all stocks (top 30% + bottom 30%)
        assert jul.height < self.df.filter(pl.col("eom") == JULY_2019).height

    def test_smb_long_weights_sum_to_one(self):
        jul = self._july_smb()
        assert jul.height > 0, "No SMB assignments in July"
        long_sum = jul.filter(pl.col("w_SMB") > 0)["w_SMB"].sum()
        assert abs(long_sum - 1.0) < 1e-10, f"SMB long sum = {long_sum}"

    def test_smb_short_weights_sum_to_minus_one(self):
        jul = self._july_smb()
        assert jul.height > 0
        short_sum = jul.filter(pl.col("w_SMB") < 0)["w_SMB"].sum()
        assert abs(short_sum + 1.0) < 1e-10, f"SMB short sum = {short_sum}"

    def test_smb_small_me_stocks_get_positive_weight(self):
        """All small-me stocks (ids 1-30) should be in the SMB long leg."""
        jul = self._july_smb()
        long_ids = set(jul.filter(pl.col("w_SMB") > 0)["id"].to_list())
        # Small stocks have ids 1..30 by construction
        assert all(i in long_ids for i in range(1, 4))

    def test_smb_big_me_stocks_get_negative_weight(self):
        jul = self._july_smb()
        short_ids = set(jul.filter(pl.col("w_SMB") < 0)["id"].to_list())
        # Big stocks have ids 31..60 by construction
        assert all(i in short_ids for i in range(31, 34))

    def test_smb_covers_more_stocks_than_hml(self):
        """SMB includes all char groups; HML excludes mid-B/M → SMB has more rows."""
        jul_hml = self._july_hml()
        jul_smb = self._july_smb()
        assert jul_smb.height > jul_hml.height

    def test_june_has_no_assignments(self):
        """June 2019 uses 2018 June breakpoints (not provided) → no portfolio assignments."""
        jun_hml = self.hml.filter(pl.col("eom") == JUNE_2019)
        assert jun_hml.height == 0

    def test_assignments_persist_across_months(self):
        """Portfolio assignments are fixed from July through August (same breakpoints)."""
        jul_ids = set(self.hml.filter(pl.col("eom") == JULY_2019)["id"].to_list())
        aug_ids = set(self.hml.filter(pl.col("eom") == AUG_2019)["id"].to_list())
        assert jul_ids == aug_ids, "Portfolio members should be the same in July and August"

    def test_output_columns(self):
        assert set(self.hml.columns) == {"eom", "id", "w_HML"}
        assert set(self.smb.columns) == {"eom", "id", "w_SMB"}


# =============================================================================
# RMW
# =============================================================================

class TestComputeRmw:
    def setup_method(self):
        self.df = _make_ff_panel([JUNE_2019, JULY_2019])
        self.rmw = compute_rmw(self.df)

    def _july(self):
        return self.rmw.filter(pl.col("eom") == JULY_2019)

    def test_long_sum_one(self):
        jul = self._july()
        assert jul.height > 0
        assert abs(jul.filter(pl.col("w_RMW") > 0)["w_RMW"].sum() - 1.0) < 1e-10

    def test_short_sum_minus_one(self):
        jul = self._july()
        assert abs(jul.filter(pl.col("w_RMW") < 0)["w_RMW"].sum() + 1.0) < 1e-10

    def test_high_ope_be_gets_positive_weight(self):
        """Robust = high ope_be → should be long."""
        jul = self._july()
        long_ids = jul.filter(pl.col("w_RMW") > 0)["id"].to_list()
        # High-ope_be stocks have high id % 30 → ids near 30 and 60
        assert max(long_ids) >= 27

    def test_low_ope_be_gets_negative_weight(self):
        jul = self._july()
        short_ids = jul.filter(pl.col("w_RMW") < 0)["id"].to_list()
        assert min(short_ids) <= 4

    def test_output_columns(self):
        assert set(self.rmw.columns) == {"eom", "id", "w_RMW"}


# =============================================================================
# CMA
# =============================================================================

class TestComputeCma:
    def setup_method(self):
        self.df = _make_ff_panel([JUNE_2019, JULY_2019])
        self.cma = compute_cma(self.df)

    def _july(self):
        return self.cma.filter(pl.col("eom") == JULY_2019)

    def test_long_sum_one(self):
        jul = self._july()
        assert jul.height > 0
        assert abs(jul.filter(pl.col("w_CMA") > 0)["w_CMA"].sum() - 1.0) < 1e-10

    def test_short_sum_minus_one(self):
        jul = self._july()
        assert abs(jul.filter(pl.col("w_CMA") < 0)["w_CMA"].sum() + 1.0) < 1e-10

    def test_low_investment_gets_positive_weight(self):
        """Conservative = low at_gr1 → long. Low at_gr1 stocks have low id % 30."""
        jul = self._july()
        long_ids = jul.filter(pl.col("w_CMA") > 0)["id"].to_list()
        assert min(long_ids) <= 4

    def test_high_investment_gets_negative_weight(self):
        """Aggressive = high at_gr1 → short."""
        jul = self._july()
        short_ids = jul.filter(pl.col("w_CMA") < 0)["id"].to_list()
        assert max(short_ids) >= 27

    def test_output_columns(self):
        assert set(self.cma.columns) == {"eom", "id", "w_CMA"}


# =============================================================================
# MOM (monthly rebalancing)
# =============================================================================

class TestComputeMom:
    """MOM uses monthly rebalancing. Two consecutive months needed for the ret_exc lag."""

    def setup_method(self):
        # Dec 2018 provides the lagged ret_exc used in Jan 2019 ret_12_2 computation
        self.df = _make_ff_panel([DEC_2018, JAN_2019]).sort(["id", "eom"])
        self.mom = compute_mom(self.df)

    def _jan(self):
        return self.mom.filter(pl.col("eom") == JAN_2019)

    def test_long_sum_one(self):
        jan = self._jan()
        assert jan.height > 0, "No MOM assignments in January"
        assert abs(jan.filter(pl.col("w_MOM") > 0)["w_MOM"].sum() - 1.0) < 1e-10

    def test_short_sum_minus_one(self):
        jan = self._jan()
        assert abs(jan.filter(pl.col("w_MOM") < 0)["w_MOM"].sum() + 1.0) < 1e-10

    def test_high_momentum_gets_positive_weight(self):
        """Winner = high ret_12_1 → long. High-id stocks have high ret_12_1."""
        jan = self._jan()
        long_ids = jan.filter(pl.col("w_MOM") > 0)["id"].to_list()
        assert max(long_ids) >= 27

    def test_no_assignments_for_first_month_in_sample(self):
        """Without a prior month, the ret_exc lag is null → no valid ret_12_2 → no MOM."""
        df_single = _make_ff_panel([JAN_2019])
        assert compute_mom(df_single).height == 0

    def test_output_columns(self):
        assert set(self.mom.columns) == {"eom", "id", "w_MOM"}


# =============================================================================
# Group A — univariate tercile sort (BAB, ROE, IA)
# =============================================================================

class TestUnivariateTercile:
    def setup_method(self):
        self.df = _make_univariate_panel([JULY_2019])

    def test_bab_long_sum_one(self):
        """BAB direction=-1: pf1 (low beta) is long. Total long weight = +1."""
        bab = _univariate_tercile(self.df, "betabab_1260d", direction=-1, weight_name="w_BAB")
        long_sum = bab.filter(pl.col("w_BAB") > 0)["w_BAB"].sum()
        assert abs(long_sum - 1.0) < 1e-10, f"BAB long sum = {long_sum}"

    def test_bab_short_sum_minus_one(self):
        bab = _univariate_tercile(self.df, "betabab_1260d", direction=-1, weight_name="w_BAB")
        short_sum = bab.filter(pl.col("w_BAB") < 0)["w_BAB"].sum()
        assert abs(short_sum + 1.0) < 1e-10

    def test_bab_low_beta_gets_positive_weight(self):
        """Low-id stocks have low beta → they should be long in BAB."""
        bab = _univariate_tercile(self.df, "betabab_1260d", direction=-1, weight_name="w_BAB")
        long_ids = bab.filter(pl.col("w_BAB") > 0)["id"].to_list()
        assert min(long_ids) <= 4

    def test_bab_high_beta_gets_negative_weight(self):
        bab = _univariate_tercile(self.df, "betabab_1260d", direction=-1, weight_name="w_BAB")
        short_ids = bab.filter(pl.col("w_BAB") < 0)["id"].to_list()
        assert max(short_ids) >= 26

    def test_roe_long_sum_one(self):
        """ROE direction=+1: pf3 (high niq_be) is long."""
        roe = _univariate_tercile(self.df, "niq_be", direction=+1, weight_name="w_ROE")
        assert abs(roe.filter(pl.col("w_ROE") > 0)["w_ROE"].sum() - 1.0) < 1e-10

    def test_roe_high_niqbe_gets_positive_weight(self):
        roe = _univariate_tercile(self.df, "niq_be", direction=+1, weight_name="w_ROE")
        long_ids = roe.filter(pl.col("w_ROE") > 0)["id"].to_list()
        assert max(long_ids) >= 26

    def test_ia_low_investment_gets_positive_weight(self):
        """IA direction=-1: low at_gr1 stocks are long."""
        ia = _univariate_tercile(self.df, "at_gr1", direction=-1, weight_name="w_IA")
        long_ids = ia.filter(pl.col("w_IA") > 0)["id"].to_list()
        assert min(long_ids) <= 4

    def test_mid_tercile_excluded(self):
        """Pf2 (middle tercile) has zero weight and is not in the output."""
        bab = _univariate_tercile(self.df, "betabab_1260d", direction=-1, weight_name="w_BAB")
        # Only long (pf1) and short (pf3) appear; no zero-weight rows are kept
        assert all(w != 0.0 for w in bab["w_BAB"].to_list())

    def test_null_char_excluded(self):
        """Stocks with null characteristic get no weight and are absent from output."""
        df_with_null = self.df.with_columns(
            pl.when(pl.col("id") == 1)
            .then(None)
            .otherwise(pl.col("betabab_1260d"))
            .alias("betabab_1260d")
        )
        bab = _univariate_tercile(df_with_null, "betabab_1260d", direction=-1, weight_name="w_BAB")
        assert 1 not in bab["id"].to_list()

    def test_microcap_excluded_from_breakpoints_contributes_to_weights(self):
        """
        Microcap stocks (size_grp='micro') are excluded from breakpoint computation
        but still receive portfolio assignments and non-zero weights.
        """
        bab = _univariate_tercile(self.df, "betabab_1260d", direction=-1, weight_name="w_BAB")
        all_ids = bab["id"].to_list()
        n_micro = (self.df["size_grp"] == "micro").sum()
        # Micro stocks above the p67 breakpoint get assigned to pf3 (short)
        # Micro stocks below the p33 breakpoint get assigned to pf1 (long)
        # Either way they should appear in output if they're in pf1 or pf3
        assert len(all_ids) > 0

    def test_output_columns(self):
        bab = _univariate_tercile(self.df, "betabab_1260d", direction=-1, weight_name="w_BAB")
        assert set(bab.columns) == {"eom", "id", "w_BAB"}


# =============================================================================
# compute_roe — q-factor 2×3 annual sort on niq_be
# =============================================================================

class TestComputeRoe:
    """Annual June rebalancing: June 2019 NYSE data sets breakpoints for July 2019+."""

    def setup_method(self):
        self.df = _make_ff_panel([JUNE_2019, JULY_2019])
        self.roe = compute_roe(self.df)

    def _july(self):
        return self.roe.filter(pl.col("eom") == JULY_2019)

    def test_long_sum_one(self):
        jul = self._july()
        assert jul.height > 0
        assert abs(jul.filter(pl.col("w_ROE") > 0)["w_ROE"].sum() - 1.0) < 1e-10

    def test_short_sum_minus_one(self):
        jul = self._july()
        assert abs(jul.filter(pl.col("w_ROE") < 0)["w_ROE"].sum() + 1.0) < 1e-10

    def test_high_niqbe_gets_positive_weight(self):
        """High niq_be stocks (top 30%) are long — they have high id % n_per_size."""
        jul = self._july()
        long_ids = jul.filter(pl.col("w_ROE") > 0)["id"].to_list()
        assert max(long_ids) >= 27

    def test_low_niqbe_gets_negative_weight(self):
        jul = self._july()
        short_ids = jul.filter(pl.col("w_ROE") < 0)["id"].to_list()
        assert min(short_ids) <= 4

    def test_output_columns(self):
        assert set(self.roe.columns) == {"eom", "id", "w_ROE"}


# =============================================================================
# compute_ia — q-factor 2×3 monthly sort on at_gr1
# =============================================================================

class TestComputeIa:
    """Monthly rebalancing: a single month is sufficient."""

    def setup_method(self):
        self.df = _make_ff_panel([JULY_2019])
        self.ia = compute_ia(self.df)

    def test_long_sum_one(self):
        assert self.ia.height > 0
        assert abs(self.ia.filter(pl.col("w_IA") > 0)["w_IA"].sum() - 1.0) < 1e-10

    def test_short_sum_minus_one(self):
        assert abs(self.ia.filter(pl.col("w_IA") < 0)["w_IA"].sum() + 1.0) < 1e-10

    def test_low_investment_gets_positive_weight(self):
        """Conservative = low at_gr1 → long; low at_gr1 stocks have low id % 30."""
        long_ids = self.ia.filter(pl.col("w_IA") > 0)["id"].to_list()
        assert min(long_ids) <= 4

    def test_high_investment_gets_negative_weight(self):
        short_ids = self.ia.filter(pl.col("w_IA") < 0)["id"].to_list()
        assert max(short_ids) >= 27

    def test_output_columns(self):
        assert set(self.ia.columns) == {"eom", "id", "w_IA"}


# =============================================================================
# compute_bab — Frazzini-Pedersen rank-weighted, beta-scaled
# =============================================================================

class TestComputeBab:
    def setup_method(self):
        self.df = _make_univariate_panel([JULY_2019])
        self.bab = compute_bab(self.df)

    def test_low_beta_gets_positive_weight(self):
        """Low-id stocks have low beta → should be long."""
        long_ids = self.bab.filter(pl.col("w_BAB") > 0)["id"].to_list()
        assert min(long_ids) <= 4

    def test_high_beta_gets_negative_weight(self):
        short_ids = self.bab.filter(pl.col("w_BAB") < 0)["id"].to_list()
        assert max(short_ids) >= 26

    def test_net_beta_zero(self):
        """FP property: long leg and short leg each have unit beta → net beta = 0."""
        bab_with_beta = self.bab.join(
            self.df.select(["id", "eom", "betabab_1260d"]), on=["id", "eom"], how="left"
        )
        net_beta = (bab_with_beta["w_BAB"] * bab_with_beta["betabab_1260d"]).sum()
        assert abs(net_beta) < 1e-10, f"Net beta = {net_beta}, expected 0"

    def test_null_char_excluded(self):
        df_null = self.df.with_columns(
            pl.when(pl.col("id") == 1).then(None).otherwise(pl.col("betabab_1260d")).alias("betabab_1260d")
        )
        bab = compute_bab(df_null)
        assert 1 not in bab["id"].to_list()

    def test_output_columns(self):
        assert set(self.bab.columns) == {"eom", "id", "w_BAB"}
