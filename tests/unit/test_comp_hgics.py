"""Regression test: comp_hgics output must not depend on the wall-clock date.

Issue #131 documented that calling `date.today()` inside `comp_hgics` made the
output depend on when the pipeline ran (each currently-active GICS record gained
one extra exploded row per day that passed between runs). The fix replaced
`date.today()` with `END_DATE` from `config.py`. This test guards against future
regressions where someone re-introduces a wall-clock dependency.

We monkeypatch the `date` name imported into `jkp.data.aux_functions` so that
`date.today()` returns two wildly different values across two consecutive
invocations of `comp_hgics`, then assert the outputs are bit-identical. If the
function ever calls `date.today()` again, the two outputs will differ and this
test will fail.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import polars as pl
import pytest

import jkp.data.aux_functions as aux_functions
from jkp.data.aux_functions import comp_hgics


def _write_minimal_hgics_fixture(raw_data_dfs: Path) -> None:
    """Write a tiny comp_hgics_na.parquet with one open-ended industry record.

    The open-ended row (`indthru` is NULL) is the trigger for the wall-clock
    fill path — without it the regression test would be trivially satisfied.
    """
    # Column is named `gics` (not `gsubind`): the upstream `comp_hgics_aux`
    # renames `gsubind` -> `gics` while building `comp_hgics_na.parquet`, so by
    # the time `comp_hgics` reads the file the column is already `gics`.
    pl.DataFrame(
        {
            "gvkey": ["001000", "001000", "001001"],
            "indfrom": [_dt.date(2010, 1, 1), _dt.date(2015, 6, 1), _dt.date(2018, 3, 1)],
            "indthru": [_dt.date(2015, 5, 31), None, None],
            "gics": [10101010, 10101020, 20202020],
        }
    ).write_parquet(raw_data_dfs / "comp_hgics_na.parquet")


def _date_subclass_returning(today_value: _dt.date) -> type:
    """Build a `date` subclass whose `today()` classmethod returns a fixed value.

    Subclassing (rather than monkeypatching the `today` method directly) keeps
    `pl.lit(date.today())` happy: Polars will still see a `datetime.date` instance.
    """

    class _FixedDate(_dt.date):
        @classmethod
        def today(cls) -> _dt.date:
            return today_value

    return _FixedDate


@pytest.mark.unit
def test_comp_hgics_independent_of_wall_clock(
    temp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """comp_hgics output must be identical regardless of `date.today()`.

    If a future change re-introduces a `date.today()` (or any other wall-clock
    read) into `comp_hgics`, the two runs below will produce different output
    because their patched "today" values differ by 6 years — a difference that
    would show up as ~2,200 extra exploded daily rows per open-ended record.
    """
    raw_data_dfs = temp_data_dir / "raw_data_dfs"
    _write_minimal_hgics_fixture(raw_data_dfs)
    monkeypatch.chdir(temp_data_dir)

    monkeypatch.setattr(aux_functions, "date", _date_subclass_returning(_dt.date(2024, 1, 15)))
    comp_hgics("national")
    first = pl.read_parquet(temp_data_dir / "na_hgics.parquet")

    (temp_data_dir / "na_hgics.parquet").unlink()

    monkeypatch.setattr(aux_functions, "date", _date_subclass_returning(_dt.date(2030, 7, 15)))
    comp_hgics("national")
    second = pl.read_parquet(temp_data_dir / "na_hgics.parquet")

    assert first.equals(second), (
        "comp_hgics output must not depend on the wall-clock date; "
        "two runs with different patched `date.today()` values produced "
        "different output, which means a wall-clock dependency has been "
        "re-introduced (regression of #131)."
    )
