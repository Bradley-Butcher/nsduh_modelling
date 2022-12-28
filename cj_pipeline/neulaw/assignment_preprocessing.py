from pathlib import Path
import pandas as pd
from tqdm import tqdm

from cj_pipeline.neulaw.load import load
from cj_pipeline.config import logger

base_path = Path(__file__).parents[2] / 'data'


def init_offence_counting(start_year: int, window: int):
    logger.info("Preparing Offence Counting...")
    harrod_county = load(base_path / 'neulaw')
    # cut off the years before the start year
    harrod_county[harrod_county["calc.year"] >= start_year]
    max_year = harrod_county["calc.year"].max()
    # add year category according to window
    def get_offence_counts(year: int):
        if year > max_year:
            raise ValueError(f"Year {year} is greater than max year {max_year}")
        logger.info(f"Counting Offences from {year} to {year + window}")
        count_df = count_year_range(harrod_county, year, year + window)
        return count_df
    return get_offence_counts


def count_year_range(df: pd.DataFrame, start_year: int, end_year:int):
    groups = ['def.gender', 'def.race', 'def.dob', 'def.uid', 'offense_category']
    year_df = df[(df['calc.year'] >= start_year) & (df['calc.year'] < end_year)]
    year_df = year_df.groupby(groups).agg({'offense_category': 'count'})
    year_df = year_df.unstack(level=-1)
    year_df = year_df.droplevel(level=0, axis=1)
    year_df = year_df.fillna(0)
    year_df["year_range"] = f"{start_year}-{end_year}"
    year_df = year_df.reset_index().rename(columns={df.index.name:'index'})
    year_df = year_df.rename(columns={'offense_category': 'index'})
    # convert to integer
    year_df["aggravated assault"] = year_df["aggravated assault"].astype(int)
    year_df["robbery"] = year_df["robbery"].astype(int)
    year_df["sex offense"] = year_df['sex offense'].astype(int)
    year_df["simple assault"] = year_df["simple assault"].astype(int)
    year_df["dui"] = year_df["dui"].astype(int)
    year_df["property"] = year_df["property"].astype(int)
    return year_df


# EXAMPLE USAGE
if __name__ == '__main__':
    offense_counts = init_offence_counting(start_year=2000, window=3)
    first_df = offense_counts(year=2000)
    second_df = offense_counts(year=2001)