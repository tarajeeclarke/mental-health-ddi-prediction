"""
Mental-Health Polypharmacy DDI Prediction
==========================================
NOTEBOOK 1 — Data Engineering & Scaling Pipeline

Loads TwoSIDES in chunks, filters to mental-health drug pairs,
creates PRR-based binary labels, and exports a clean subset CSV
for downstream modeling.

Author: tarajeeclarke
Course: HADM 283 - AI in Healthcare, Hofstra University
"""

import pandas as pd
import polars as pl
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. MENTAL-HEALTH DRUG KEYWORD LIST
# ─────────────────────────────────────────────────────────────

MENTAL_HEALTH_KEYWORDS = [
    # SSRIs / SNRIs
    "fluoxetine", "sertraline", "escitalopram", "citalopram", "paroxetine",
    "fluvoxamine", "venlafaxine", "duloxetine", "desvenlafaxine", "levomilnacipran",
    # Antidepressants (other)
    "bupropion", "mirtazapine", "trazodone", "nefazodone", "amitriptyline",
    "nortriptyline", "imipramine", "clomipramine",
    # Antipsychotics
    "quetiapine", "olanzapine", "risperidone", "aripiprazole", "ziprasidone",
    "haloperidol", "clozapine", "lurasidone", "paliperidone", "asenapine",
    # Mood stabilizers
    "lithium", "valproate", "valproic acid", "lamotrigine", "carbamazepine",
    "oxcarbazepine",
    # Benzodiazepines
    "alprazolam", "diazepam", "clonazepam", "lorazepam", "temazepam",
    "oxazepam", "triazolam", "midazolam",
    # Stimulants
    "methylphenidate", "amphetamine", "lisdexamfetamine", "dextroamphetamine",
    "atomoxetine", "modafinil",
    # Sleep / other
    "zolpidem", "eszopiclone", "buspirone", "hydroxyzine",
]


def matches_mental_health(drug_name: str) -> bool:
    """Return True if the drug name contains any mental-health keyword."""
    if pd.isna(drug_name):
        return False
    drug_lower = str(drug_name).lower()
    return any(kw in drug_lower for kw in MENTAL_HEALTH_KEYWORDS)


# ─────────────────────────────────────────────────────────────
# 2. STEP A — LOAD 100,000-ROW CHUNK FROM TWOSIDES
# ─────────────────────────────────────────────────────────────

def load_twosides_chunk(filepath: str, nrows: int = 100_000) -> str:
    """
    Load first N rows of TwoSIDES and save as intermediate plain CSV.
    The intermediate file avoids encoding issues when reading into Polars.
    
    Returns path to the saved intermediate CSV.
    """
    print(f"Loading {nrows:,} rows from TwoSIDES...")

    df_chunk = pd.read_csv(
        filepath,
        nrows=nrows,
        low_memory=False,
    )

    print(f"  Loaded shape: {df_chunk.shape}")
    print(f"  Columns: {list(df_chunk.columns)}")

    intermediate_path = "twosides_chunk_100k.csv"
    df_chunk.to_csv(intermediate_path, index=False)
    print(f"  Saved intermediate file: {intermediate_path}")
    return intermediate_path


# ─────────────────────────────────────────────────────────────
# 3. STEP B — LOAD CHUNK INTO POLARS (relaxed parsing)
# ─────────────────────────────────────────────────────────────

def load_into_polars(csv_path: str) -> pl.DataFrame:
    """
    Read the intermediate CSV into Polars with relaxed parsing settings
    to handle encoding irregularities and ragged rows.
    """
    print("\nLoading chunk into Polars...")

    df = pl.read_csv(
        csv_path,
        encoding="utf8-lossy",
        ignore_errors=True,
        truncate_ragged_lines=True,
        infer_schema_length=10_000,
    )

    print(f"  Polars shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# 4. STEP C — FILTER TO MENTAL-HEALTH DRUG PAIRS
# ─────────────────────────────────────────────────────────────

def filter_mental_health(df: pl.DataFrame) -> pl.DataFrame:
    """
    Keep rows where drug_1 OR drug_2 matches the mental-health keyword list.
    Converts to pandas for filtering, then back to Polars for consistency.
    """
    print("\nFiltering to mental-health drug pairs...")

    df_pd = df.to_pandas()

    mask = (
        df_pd["drug_1_concept_name"].apply(matches_mental_health) |
        df_pd["drug_2_concept_name"].apply(matches_mental_health)
    )

    df_mental = df_pd[mask].copy()
    print(f"  Rows after mental-health filter: {len(df_mental):,}")
    print(f"  Unique drug_1 values: {df_mental['drug_1_concept_name'].nunique()}")
    print(f"  Unique drug_2 values: {df_mental['drug_2_concept_name'].nunique()}")

    return pl.from_pandas(df_mental)


# ─────────────────────────────────────────────────────────────
# 5. STEP D — PRR CONVERSION + BINARY LABEL CREATION
# ─────────────────────────────────────────────────────────────

def create_labels(df: pl.DataFrame) -> pd.DataFrame:
    """
    Convert PRR to numeric and create binary label:
      - PRR > 1  → label = 1 (elevated signal)
      - PRR ≤ 1  → label = 0 (no signal)
    """
    print("\nCreating PRR-based binary labels...")

    df_pd = df.to_pandas()

    # Safe numeric conversion — coerce unparseable values to NaN
    df_pd["PRR_float"] = pd.to_numeric(df_pd["PRR"], errors="coerce")

    # Drop rows where PRR could not be parsed
    before = len(df_pd)
    df_pd.dropna(subset=["PRR_float"], inplace=True)
    after = len(df_pd)
    print(f"  Dropped {before - after} rows with unparseable PRR values")

    # Binary label
    df_pd["label"] = (df_pd["PRR_float"] > 1).astype(int)

    label_counts = df_pd["label"].value_counts()
    print(f"  Label 1 (signal):    {label_counts.get(1, 0):,}")
    print(f"  Label 0 (no signal): {label_counts.get(0, 0):,}")
    print(f"  Class imbalance ratio: {label_counts.get(1, 0) / max(label_counts.get(0, 1), 1):.1f}:1")

    return df_pd


# ─────────────────────────────────────────────────────────────
# 6. STEP E — EXPORT SUBSET
# ─────────────────────────────────────────────────────────────

def export_subset(df: pd.DataFrame, output_path: str = "twosides_mental_health_subset.csv"):
    """Save the prepared mental-health subset for use in the modeling notebook."""
    df.to_csv(output_path, index=False)
    print(f"\nSubset exported: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TwoSIDES — Mental-Health DDI Data Engineering Pipeline")
    print("=" * 60)

    # ── Provide path to TwoSIDES ──────────────────────────────
    # Download from: https://tatonettilab.org/resources/nsides/
    # Expected file: 2018-06-10.paired-t.csv.xz (or uncompressed .csv)
    twosides_path = "twosides.csv"

    if not os.path.exists(twosides_path):
        print(
            f"\n[ERROR] '{twosides_path}' not found.\n"
            "Download TwoSIDES from: https://tatonettilab.org/resources/nsides/\n"
            "Rename the file to 'twosides.csv' and place it in this directory.\n"
            "If using the .xz compressed version, decompress first:\n"
            "  python -c \"import lzma, shutil; "
            "shutil.copyfileobj(lzma.open('2018-06-10.paired-t.csv.xz'), "
            "open('twosides.csv', 'wb'))\""
        )
        return

    # Run pipeline
    intermediate = load_twosides_chunk(twosides_path, nrows=100_000)
    df_polars    = load_into_polars(intermediate)
    df_filtered  = filter_mental_health(df_polars)
    df_labeled   = create_labels(df_filtered)
    export_subset(df_labeled)

    print("\nData engineering pipeline complete.")
    print("Next: run 02_modeling.py (or open 02_modeling.ipynb) for ML modeling.")


if __name__ == "__main__":
    main()
