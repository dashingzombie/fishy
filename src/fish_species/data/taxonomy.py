"""Fish genus/species label derivation."""

from __future__ import annotations

import pandas as pd

def derive_genus_from_species(series: pd.Series) -> pd.Series:
    """Return the first token of each valid binomial species label."""
    cleaned = series.astype("string").str.strip()
    genus = cleaned.str.split().str[0]
    return genus.mask(cleaned.isna() | ~cleaned.str.contains(r"\s", regex=True))


def add_fish_taxonomy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Normalize configured species labels and derive missing genus labels."""
    result = df.copy()
    target_cols = cfg["data"]["target_cols"]
    species_col = target_cols.get("species", "species_label")
    genus_col = target_cols.get("genus", "genus")
    if species_col not in result.columns:
        raise ValueError(f"Fish species column not found: {species_col}")

    result[species_col] = result[species_col].astype("string").str.strip()
    inferred = derive_genus_from_species(result[species_col])
    if genus_col not in result.columns:
        result[genus_col] = inferred
    else:
        result[genus_col] = result[genus_col].astype("string").str.strip().fillna(inferred)

    result["__taxon_for_split__"] = result[species_col].where(
        result[species_col].notna(), result[genus_col]
    )
    return result


__all__ = ["add_fish_taxonomy", "derive_genus_from_species"]
