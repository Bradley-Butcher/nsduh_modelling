from pathlib import Path
import pandas as pd

from functools import reduce, lru_cache
from cj_pipeline.neulaw.load import load as load_neulaw
from cj_pipeline.config import logger, CRIMES, CRIMES_GROUP, DEMOGRAPHICS

base_path = Path(__file__).parents[2] / 'data'


def init_neulaw(start_year: int, window: int):
  logger.info("Preparing offence counting ...")

  df = _get_neulaw(start_year)
  max_year = df['calc.year'].max()

  # add year category according to window
  def get_entries(year: int):
    logger.info(f'Extracting Neulaw for years {year}-{year + window}')
    _check_year_validity(year, max_year=max_year, window=window)
    years_df = _preprocess_neulaw(df, start_year=year, end_year=year + window)
    return years_df

  return get_entries, max_year


def init_ncvs(
    start_year: int,
    window: int,
    smoothing: str,
    rate_mult: dict = None,
):
  logger.info('Preparing NCVS record extraction ...')
  ncvs = pd.read_csv(base_path / 'processed' / f'ncvs_{smoothing}.csv')
  ncvs = ncvs[ncvs['offender_age'] != '< 18']
  ncvs = ncvs[ncvs['ncvs_year'] >= start_year]
  max_year = ncvs['ncvs_year'].max()

  # computes lambdas on all of neulaw
  neulaw = _get_neulaw(start_year)
  neulaw = _preprocess_neulaw(
    neulaw, start_year=start_year, end_year=neulaw['calc.year'].max())
  lambdas = _ncvs_crime_lambdas(neulaw)

  def get_entries(year: int):
    _check_year_validity(year, max_year=max_year, window=window)
    years_df = ncvs.query(f'{year} <= ncvs_year <= {year + window}')
    years_df = years_df.groupby(CRIMES_GROUP, as_index=False).agg({
      'arrest_rate': 'mean', 'arrest_rate_smooth': 'mean',  # avg rate in window
      'reporting_rate': 'mean', 'count': 'sum'
    })
    years_df = pd.merge(years_df, lambdas, how='left', on=CRIMES_GROUP)
    _adjust_rates(years_df, rate_mult)  # inplace
    return years_df

  return get_entries, max_year


def init_nsduh(
    start_year: int,
    window: int,
    smoothing: str,
    rate_mult: dict = None,
):
  logger.info('Preparing NSDUH record counting ...')

  nsduh = pd.read_csv(base_path / 'processed' / f'nsduh_{smoothing}.csv')
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

    nsduh_crimes = ['dui', 'drugs_sell', 'drugs_use']
    arrests = pd.concat([
      _melt(crime=c, suffix='_ar', col_name='arrest_rate')
      for c in nsduh_crimes
    ])
    smoothed_arrests = pd.concat([
      _melt(crime=c, suffix='_sar', col_name='arrest_rate_smooth')
      for c in nsduh_crimes
    ])
    lambdas = pd.concat([
      _melt(crime=c, suffix='_lam_ar', col_name='lambda')
      for c in nsduh_crimes
    ])
    smoothed_lambdas = pd.concat([
      _melt(crime=c, suffix='_lam_sar', col_name='lambda_smooth')
      for c in nsduh_crimes
    ])

    years_df = reduce(
      lambda df0, df1: pd.merge(df0, df1, on=CRIMES_GROUP),
      [arrests, smoothed_arrests, lambdas, smoothed_lambdas]
    )
    _adjust_rates(years_df, rate_mult)  # inplace
    return years_df

  return get_entries, max_year


def _check_year_validity(year: int, max_year: int, window: int):
  if year > max_year:
    raise ValueError(f"Year {year} is greater than max year {max_year}")
  if year + window > max_year:
    logger.warning(f"Year {year + window} is greater than max year {max_year}")


def _adjust_rates(df, rate_mult):
  if rate_mult is not None:
    for race, mult in rate_mult.items():
      idx = df['offender_race'] == race
      df.loc[idx, 'arrest_rate'] *= mult
      df.loc[idx, 'arrest_rate_smooth'] *= mult
    df['arrest_rate'].clip(0, 1, inplace=True)
    df['arrest_rate_smooth'].clip(0, 1, inplace=True)


@lru_cache
def _get_neulaw(start_year: int) -> pd.DataFrame:
  df = load_neulaw(base_path / 'neulaw')

  # subset year and handle special column values
  df = df[df['calc.year'] >= start_year]
  df = df[df['def.gender'].isin(('Female', 'Male'))]
  df = df[df['calc.race'].isin(('Black', 'White', 'Hispanic'))]
  df = df[df['def.race'].isin(('Black', 'White'))]

  return df


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


def _ncvs_crime_lambdas(neulaw):
  lambdas = []
  neulaw['n_crimes'] = neulaw[CRIMES].sum(axis=1)
  for crime in CRIMES:
    grouped = neulaw[neulaw[crime] > 0].groupby(DEMOGRAPHICS)
    lam = grouped.apply(lambda g: (g['n_crimes'] > 1).mean())
    lambdas.append(lam.to_frame(crime).reset_index())

  lambdas = reduce(
    lambda df0, df1: pd.merge(df0, df1, on=DEMOGRAPHICS, how='left'), lambdas)
  lambdas = lambdas.melt(
    id_vars=DEMOGRAPHICS, value_vars=CRIMES,
    var_name='crime_recode', value_name='lambda')
  lambdas.rename(
    columns={
      'calc.race': 'offender_race',
      'def.gender': 'offender_sex',
      'age_cat': 'offender_age',
    },
    inplace=True,
  )

  return lambdas


if __name__ == '__main__':
  neulaw_gen, _ = init_neulaw(start_year=1992, window=20)

  years_df = neulaw_gen(1992)
  lambdas = _ncvs_crime_lambdas(years_df)
  lambdas.to_csv(base_path / 'processed' / 'crime_lambdas.csv', index=False)
