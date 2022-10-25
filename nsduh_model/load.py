import pandas as pd
from pathlib import Path
from nsduh_model.preprocessing import preprocess, get_variables

data_path = Path(__file__).parents[1] / 'data'
years = [2019]


def read_txt(year: int, max_rows: int):
    year_data_path = data_path / f'NSDUH_{year}_Tab.txt'
    kwargs = {}
    if max_rows > 0:
        kwargs['nrows'] = max_rows
    df = pd.read_csv(year_data_path, sep='\t', usecols=get_variables(), **kwargs)
    df["Year"] = year
    return df


def load_nsduh(max_rows: int = -1):
    df = pd.concat([read_txt(year, max_rows) for year in years])
    return preprocess(df)
