import pandas as pd
from pathlib import Path
from cj_pipeline.config import logger


def load(base_path: Path):
    logger.info(f"Loading data from {base_path}")
    hc = _load_hc(base_path)
    logger.info(f"Loaded {len(hc)} rows from {base_path}")
    logger.info(f"Merging offense categories")
    hc = merge_offense_categories(base_path, hc)
    logger.info(f"Done loading.")
    return hc


def merge_offense_categories(base_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    category_csv = pd.read_csv(base_path / 'neulaw_offensecat.csv')
    df = df.merge(category_csv, on='calc.detailed', how='left')
    return df


def _load_hc(base_path: Path, sample_idxs: list[int] | None = None) -> pd.DataFrame:
    hc_path = base_path / 'hc.csv'
    if sample_idxs is not None:
        return pd.read_csv(hc_path).iloc[sample_idxs]
    else:
        hc = pd.read_csv(hc_path, low_memory=True, skiprows=range(1, 1_200_000))
    return hc