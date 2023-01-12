import pandas as pd
from pathlib import Path
from cj_pipeline.nsduh.preprocess import get_variables

years = range(1992, 2020)
stata_years = [1999, 2000, 2001]
data_path = Path(__file__).parents[2] / 'data' / 'nsduh'


def read_nsduh(year: int, max_rows: int = None) -> pd.DataFrame:
  if year in stata_years:
    path = data_path / f'NHSDA-{year}-DS0001-data-stata.dta'
    kwargs = {'iterator': True, 'chunksize': max_rows}
    reader = pd.read_stata(path, **kwargs)  # used to get available var names
    kwargs['columns'] = list(
      get_variables().intersection(reader.variable_labels().keys()))
    df = next(pd.read_stata(path, **kwargs))   # take just the first max_rows
  else:
    path = data_path / f'NSDUH_{year}_Tab.txt'
    kwargs = {'nrows': max_rows} if max_rows is not None else {}
    df = pd.read_csv(
      path, sep='\t', usecols=lambda c: c in get_variables(), **kwargs)

  df['YEAR'] = year
  return df


def load_nsduh(max_rows: int = None) -> pd.DataFrame:
  df = pd.concat([read_nsduh(year, max_rows) for year in years])
  return df
