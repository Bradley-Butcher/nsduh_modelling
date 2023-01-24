from pathlib import Path
import pandas as pd

from cj_pipeline.config import logger, CRIMES, CRIMES_GROUP
from cj_pipeline.neulaw.load import load as load_neulaw

base_path = Path(__file__).parents[2] / 'data'


def init_neulaw(
    start_year: int,
    window: int,
    melt: bool = False,
):
  logger.info("Preparing offence counting ...")
  df = load_neulaw(base_path / 'neulaw')

  # subset year and handle special column values
  df = df[df['calc.year'] >= start_year]
  df = df[df['def.gender'].isin(('Female', 'Male'))]
  df = df[df['calc.race'].isin(('Black', 'White', 'Hispanic'))]
  df = df[df['def.race'].isin(('Black', 'White'))]
  max_year = df['calc.year'].max()

  # add year category according to window
  def get_entries(year: int):
    logger.info(f'Extracting Neulaw for years {year}-{year + window}')
    _check_year_validity(year, max_year=max_year, window=window)

    years_df = _preprocess_neulaw(df, start_year=year, end_year=year + window)
    if melt:
      # convert from wide to tall
      years_df = years_df.melt(
        id_vars=years_df.columns.difference(CRIMES), value_vars=CRIMES,
        var_name='offense_category', value_name='offense_count')
    return years_df

  return get_entries, max_year


def init_ncvs(start_year: int, window: int):
  logger.info('Preparing NCVS record extraction ...')
  ncvs = pd.read_csv(base_path / 'processed' / 'ncvs.csv')
  ncvs = ncvs[ncvs['offender_age'] != '< 18']
  ncvs = ncvs[ncvs['ncvs_year'] >= start_year]
  max_year = ncvs['ncvs_year'].max()

  def get_entries(year: int):
    _check_year_validity(year, max_year=max_year, window=window)
    years_df = ncvs.query(f'{year} <= ncvs_year <= {year + window}')
    years_df = years_df.groupby(CRIMES_GROUP, as_index=False).agg({
      'arrest_rate': 'mean', 'arrest_rate_smooth': 'mean',  # avg rate in window
      'reporting_rate': 'mean', 'count': 'sum'
    })
    return years_df

  return get_entries, max_year


def init_nsduh(start_year: int, window: int):
  logger.info('Preparing NSDUH record counting ...')

  nsduh = pd.read_csv(base_path / 'processed' / 'nsduh.csv')
  nsduh = nsduh[nsduh['offender_age'] != '< 18']
  nsduh = nsduh[nsduh['YEAR'] >= start_year]
  max_year = nsduh['YEAR'].max()

  def get_entries(year: int):
    _check_year_validity(year, max_year=max_year, window=window)
    years_df = nsduh.query(f'{year} <= YEAR <= {year + window}')

    groups = [var for var in CRIMES_GROUP if var != 'crime_recode']
    arrest_cols = years_df.columns.difference(groups + ['YEAR', 'count'])
    agg_fns = {var: 'mean' for var in arrest_cols}
    agg_fns['count'] = 'sum'
    years_df = years_df.groupby(groups, as_index=False).agg(agg_fns)

    def _melt(crime, suffix, col_name):
      melted = years_df.melt(
        id_vars=groups, value_vars=crime + suffix,
        var_name='crime_recode', value_name=col_name)
      melted = melted.replace({'crime_recode': {crime + suffix: crime}})
      return melted

    arrests = pd.concat([
      _melt(crime='dui', suffix='_ar', col_name='arrest_rate'),
      _melt(crime='drugs_sell', suffix='_ar', col_name='arrest_rate'),
      _melt(crime='drugs_use', suffix='_ar', col_name='arrest_rate'),
    ])
    smoothed_arrests = pd.concat([
      _melt(crime='dui', suffix='_sar', col_name='arrest_rate_smooth'),
      _melt(crime='drugs_sell', suffix='_sar', col_name='arrest_rate_smooth'),
      _melt(crime='drugs_use', suffix='_sar', col_name='arrest_rate_smooth'),
    ])
    years_df = pd.merge(arrests, smoothed_arrests, on=CRIMES_GROUP)

    return years_df

  return get_entries, max_year


def _check_year_validity(year: int, max_year: int, window: int):
  if year > max_year:
    raise ValueError(f"Year {year} is greater than max year {max_year}")
  if year + window > max_year:
    logger.warning(f"Year {year + window} is greater than max year {max_year}")


def _preprocess_neulaw(df: pd.DataFrame, start_year: int, end_year: int):
  groups = ['def.gender', 'calc.race', 'def.race', 'def.dob', 'def.uid', 'offense_category']

  year_df = df.query(f'{start_year} <= `calc.year` <= {end_year}')
  year_df = year_df.groupby(groups).agg({'offense_category': 'count'})
  year_df = year_df.unstack(level=-1)
  year_df = year_df.droplevel(level=0, axis=1)
  year_df = year_df.fillna(0)

  year_df['year_range'] = f'{start_year}-{end_year}'
  year_df = year_df.reset_index().rename(columns={df.index.name: 'index'})
  year_df = year_df.rename(columns={'offense_category': 'index'})

  # add age within the time-frame
  age_year = pd.to_datetime(str(end_year))  # age at which all crimes were commited
  year_df['age'] = age_year - pd.to_datetime(year_df['def.dob'])
  year_df['age'] = year_df['age'].dt.days / 365.25
  year_df = year_df[year_df['age'] > 10]  # likely data entry errors
  year_df['age_nsduh'] = pd.cut(
    year_df['age'], right=True,
    bins=[0, 17, 34, 500], labels=['< 18', '18-34', '> 34']).astype('str')
  year_df['age_ncvs'] = pd.cut(
    year_df['age'], right=True,
    bins=[0, 17, 29, 500], labels=['< 18', '18-29', '> 29']).astype('str')
  year_df['age_cat'] = year_df['age_ncvs']  # for now take as default

  # remove all underage entries
  year_df = year_df[year_df['age_cat'] != '< 18']

  # convert to integer
  for col in CRIMES:
    year_df[col] = year_df[col].astype(int)

  return year_df


# EXAMPLE USAGE
if __name__ == '__main__':
  offense_counts, _ = init_neulaw(start_year=2000, window=3)
  first_df = offense_counts(year=2000)
  second_df = offense_counts(year=2001)