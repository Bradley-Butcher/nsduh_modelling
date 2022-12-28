from pathlib import Path
import pandas as pd
from tqdm import tqdm

from cj_pipeline.neulaw.load import load
from cj_pipeline.config import logger

base_path = Path(__file__).parents[2] / 'data'

def get_offence_counts(start_year: int, window: int):
    harrod_county = load(base_path / 'neulaw')
    # cut off the years before the start year
    harrod_county[harrod_county["calc.year"] >= start_year]
    max_year = harrod_county["calc.year"].max()
    # add year category according to window
    start_windows = range(start_year, max_year - window + 1)
    end_windows = range(start_year + window, max_year + 1)
    logger.info("Starting Offence Counting")
    for start_window, end_window in zip(start_windows, end_windows):
        logger.info(f"Counting Offences from {start_window} to {end_window}")
        count_df = count_year_range(harrod_county, start_window, end_window)
        count_df.to_csv(base_path / 'processed' / f'offence_counts_{start_window}_{end_window}.csv')


def count_year_range(df: pd.DataFrame, start_year: int, end_year:int):
    groups = ['def.gender', 'def.race', 'def.uid', 'offense_category']
    year_df = df[(df['calc.year'] >= start_year) & (df['calc.year'] < end_year)]
    year_df = year_df.groupby(groups).agg({'offense_category': 'count'})
    year_df = year_df.unstack(level=-1)
    year_df = year_df.fillna(0)
    year_df["year_range"] = f"{start_year}-{end_year}"
    return year_df


if __name__ == '__main__':
    get_offence_counts(2000, 5)