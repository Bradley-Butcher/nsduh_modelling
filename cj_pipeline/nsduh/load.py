import pandas as pd
from pathlib import Path
from cj_pipeline.nsduh.preprocess import get_variables

data_path = Path(__file__).parents[2] / 'data' / 'nsduh'
# years = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
years = list(range(1992, 2020))
stata_years = [1999, 2000, 2001]


def read_nsduh(year: int, max_rows: int) -> pd.DataFrame:
    if year in stata_years:
      year_data_path = data_path / f'NHSDA-{year}-DS0001-data-stata.dta'
      if max_rows > 0:
        reader = pd.read_stata(year_data_path, chunksize=max_rows)
        df = next(reader)  # take just the first max_rows
      else:
        df = pd.read_stata(year_data_path)
    else:
      year_data_path = data_path / f'NSDUH_{year}_Tab.txt'
      kwargs = {'nrows': max_rows} if max_rows > 0 else {}
      df = pd.read_csv(
        year_data_path, sep='\t',
        usecols=lambda c: c in get_variables(),
        **kwargs
      )

    df['YEAR'] = year
    return df


def load_nsduh(max_rows: int = -1) -> pd.DataFrame:
    df = pd.concat([read_nsduh(year, max_rows) for year in years])
    return df
