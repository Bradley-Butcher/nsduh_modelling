import pandas as pd
from cj_pipeline.config import logger

from pathlib import Path

data_path = Path(__file__).parents[2] / 'data' / 'ncvs'


def rename(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = {
        'toc_code_new_ncvs': 'crime_type',
    }
    df = df.rename(columns=rename_dict)
    # drop everything except that just renamed
    return df

def load() -> pd.DataFrame:
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path / 'ncvs.csv')
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    logger.info(f"Renaming columns")
    df = rename(df)
    return df