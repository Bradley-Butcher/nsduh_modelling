import pandas as pd
from pathlib import Path
from cj_pipeline.nsduh.preprocessing import preprocess, get_variables

data_path = Path(__file__).parents[2] / 'data' / 'nsduh'
years = [2013, 2014, 2015, 2016, 2017, 2018, 2019]


def read_txt(year: int, max_rows: int) -> pd.DataFrame:
    year_data_path = data_path / f'NSDUH_{year}_Tab.txt'
    kwargs = {}
    if max_rows > 0:
        kwargs['nrows'] = max_rows
    df = pd.read_csv(
      year_data_path, sep='\t',
      usecols=lambda c: c in get_variables(), **kwargs)
    df["Year"] = year
    return df


def load_nsduh(max_rows: int = -1) -> pd.DataFrame:
    df = pd.concat([read_txt(year, max_rows) for year in years])
    return preprocess(df)
