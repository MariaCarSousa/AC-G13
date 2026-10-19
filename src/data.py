"""Shared loader for the hockey CSVs.

Usage (from a notebook in notebooks/):
    import sys; sys.path.append("..")
    from src.data import load_table, load_all

    teams = load_table("Teams")
    tables = load_all()          # dict: table name -> DataFrame
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "csv_datas"

# Columns that look numeric but are identifiers or codes, so they stay as text.
TEXT_COLUMNS = {
    "playerID", "coachID", "hofID", "legendsID", "ihdbID", "hrefID",
    "tmID", "franchID", "lgID", "confID", "divID", "oppID",
    "tmIDWinner", "tmIDLoser", "lgIDWinner", "lgIDLoser",
    "IDgoalie1", "IDgoalie2", "ID", "Code",
}


def _convert_column(col: pd.Series) -> pd.Series:
    """Turn a text column into numbers when every non-empty value is numeric."""
    values = col.str.strip().replace("", pd.NA)
    non_missing = values.dropna()
    if non_missing.empty:
        return values

    numbers = pd.to_numeric(non_missing, errors="coerce")
    if numbers.isna().any():
        return values  # genuine text column (names, notes, positions, ...)

    numbers = pd.to_numeric(values)
    if (numbers.dropna() % 1 == 0).all():
        return numbers.astype("Int64")  # nullable integer: keeps missing values
    return numbers.astype("Float64")


def load_table(name: str) -> pd.DataFrame:
    """Load one table by name (e.g. "Teams") with proper column types."""
    path = DATA_DIR / f"{name}.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in df.columns:
        if c in TEXT_COLUMNS:
            df[c] = df[c].str.strip().replace("", pd.NA)
        else:
            df[c] = _convert_column(df[c])
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Load every CSV in csv_datas/ into a dict keyed by table name."""
    return {p.stem: load_table(p.stem) for p in sorted(DATA_DIR.glob("*.csv"))}
