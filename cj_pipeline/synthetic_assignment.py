import numpy as np
import pandas as pd
from functools import partial
from collections import Counter

from cj_pipeline.config import BASE_DIR, CRIMES, CRIMES_GROUP, NEULAW_TO_NCVS, NEULAW_TO_NSDUH, logger
from cj_pipeline.neulaw.assignment_preprocessing import init_neulaw, init_ncvs, init_nsduh


def get_synth(
    start_year: int, end_year: int, window: int, seed: int = 0,
    lam: float = 1, omega: float = 1
) -> pd.DataFrame:
  arrest_col = 'arrest_rate_smooth'
  file_path = _file_path(
    start_year=start_year, end_year=end_year, window=window,
    seed=seed, lam=lam, omega=omega)

  logger.info(f'Loading synth assignments {start_year}-{end_year} ({window})')
  if file_path.is_file():
    logger.info('Loading from the disk')
    df = pd.read_csv(file_path)
  else:
    logger.info('Generating the synthetic data')
    df = rolling_crime_assignment(
      start_year=start_year, end_year=end_year, window=window, seed=seed,
      lam=lam, omega=omega, arrest_col=arrest_col)
    df.to_csv(file_path, index=False)

  return df


def rolling_crime_assignment(
    start_year: int, end_year: int, window: int, seed: int, **kwargs
) -> pd.DataFrame:
  rng = np.random.RandomState(seed=seed)
  sample_window = _window_sampler(
    start_year=start_year, end_year=end_year, window=window, rng=rng, **kwargs)
  years_in_window = window + 1  # end_year (= start_year + window) is included

  samples = []
  for idx, window_end in enumerate(range(start_year + window, end_year + 1)):
    logger.info(f'Sampling for year window ending by year {window_end}')
    samples.append(sample_window(
      window_end, n_samples_div=1 if idx == 0 else years_in_window))

  df = pd.concat(samples)
  df = df.drop(columns=['age_cat'])  # may conflict as people age between windows
  df = df.groupby(df.columns.difference(CRIMES).to_list()).sum().reset_index()
  df = _add_age(df, end_year=end_year)

  return df


def _file_path(start_year, end_year, window, lam, omega, seed):
  file_name = f'lam{lam:.2f}_om{omega:.2f}-{seed}.csv'
  data_path = BASE_DIR / 'data' / 'scratch' / 'synth'
  data_path /= f'{start_year}-{end_year}_{window}'
  data_path.mkdir(parents=True, exist_ok=True)
  return data_path / file_name


def _add_age(df, end_year):  # TODO: code duplication with assignment_preprocessing.py
  age = pd.to_datetime(str(end_year)) - pd.to_datetime(df['def.dob'])
  age = age.dt.days / 365.25

  df = df[age > 10]  # likely data entry errors
  df['age_cat'] = pd.cut(
    age, right=True, bins=[0, 17, 29, 500],
    labels=['< 18', '18-29', '> 29']).astype('str')
  df = df[df['age_cat'] != '< 18']  # remove all underage entries

  return df


def _count_unobserved(pop, lam, arrest_col):
  total_crimes = lam * pop['offense_count'] / pop[arrest_col]
  pop['total_crimes'] = np.where(
    pop[arrest_col] > 0,  # TODO: add warning when this is False
    total_crimes, pop['offense_count']
  ).round().astype('int')
  pop['unobserved_crimes'] = pop['total_crimes'] - pop['offense_count']
  pop['unobserved_per_person'] = pop['unobserved_crimes'] / pop['pop_size']
  return pop


def _sample_unobserved(df, groups, n_samples_div, rng):
  def _sample(group):
    if group['unobserved_crimes'].nunique() > 1:
      raise ValueError(f'Conflicting no. of unobserved crimes to be generated '
                       f'{group["unobserved_crimes"].unique()}')

    n_samples = int(group['unobserved_crimes'].mean() / n_samples_div)
    if group['crime_weight'].sum() <= 0 or n_samples < 1:
      return None  # no crimes of this type (happens for some < 18 categories)

    samples = group['def.uid'].sample(
      n=n_samples,
      replace=True,
      weights=group['crime_weight'],
      random_state=rng
    )
    return list(Counter(samples).items())

  # df is melted over crimes -> subset only individual records for `crime`
  samples = df.groupby(groups).apply(_sample).to_frame('def.uid').reset_index()
  samples = samples[samples['def.uid'].notna()]  # remove categories w/o samples
  samples = samples.explode('def.uid')
  samples['offense_unobserved'] = samples['def.uid'].str[1]
  samples['def.uid'] = samples['def.uid'].str[0]

  return samples


def _add_unobserved(
    df, group_all, crimes, lam, omega, arrest_col, n_samples_div, rng):
  groups = [col for col in group_all if col != 'offense_category']

  offenses = df.groupby(group_all, as_index=False)['offense_count'].sum()
  offenses = pd.merge(
    offenses,
    df.groupby(groups)['def.uid'].nunique().to_frame('pop_size').reset_index(),
    how='left', on=groups)
  offenses = pd.merge(
    offenses, crimes, how='left', left_on=group_all, right_on=CRIMES_GROUP)
  offenses = offenses.drop(columns=CRIMES_GROUP)  # de-duplicate columns

  # compute sampling weights
  offenses = _count_unobserved(offenses, lam=lam, arrest_col=arrest_col)
  if offenses['unobserved_per_person'].isna().sum() > 0:
    raise RuntimeError('Failed to assign unobserved offenses')
  unobs_cols = lambda c: c + ['unobserved_per_person', 'unobserved_crimes']
  df = pd.merge(df, offenses[unobs_cols(group_all)], how='left', on=group_all)
  df['crime_weight'] = df['unobserved_per_person'] + omega * df['offense_count']

  # sample unobserved
  samples = _sample_unobserved(
    df=df, groups=group_all, n_samples_div=n_samples_div, rng=rng)
  df = pd.merge(df, samples, how='left', on=group_all + ['def.uid'])
  df['offense_unobserved'] = df['offense_unobserved'].fillna(0).astype('int')
  df['offense_total'] = df['offense_count'] + df['offense_unobserved']

  return df


def _window_sampler(
    start_year, end_year, window, lam, omega, arrest_col, rng):
  # load data for given time-frame
  ncvs_gen, _ = init_ncvs(start_year, window=window)
  nsduh_gen, _ = init_nsduh(start_year, window=window)
  neulaw_gen, _ = init_neulaw(start_year, window=window, melt=True)

  def _window(window_end: int, n_samples_div: float = 1.0):
    year = window_end - window
    if not start_year <= year <= end_year:
      raise ValueError(
        f'Last year {window_end} not compatible with start year {start_year} '
        f'and window {window}.')

    # load data for given time-frame
    df = neulaw_gen(year)
    ncvs = ncvs_gen(year)
    nsduh = nsduh_gen(year)

    # sample new unobserved crimes
    len_before = len(df)
    nsduh_ids = df['offense_category'].isin(['dui', 'drugs_use', 'drugs_sell'])
    _sample = partial(
      _add_unobserved, lam=lam, omega=omega, arrest_col=arrest_col,
      n_samples_div=n_samples_div, rng=rng
    )
    df = pd.concat([
      _sample(df=df[~nsduh_ids], group_all=NEULAW_TO_NCVS, crimes=ncvs),
      _sample(df=df[nsduh_ids], group_all=NEULAW_TO_NSDUH, crimes=nsduh),
    ])
    if len(df) != len_before:
      raise RuntimeError('Bug: sampling of unobserved changed size of the data')

    # convert back into the wide format
    df = pd.pivot_table(
      df, columns='offense_category', values='offense_total', sort=False,
      index=['def.gender', 'calc.race', 'def.uid', 'def.dob', 'age_cat']
    ).reset_index()

    return df

  return _window


def main():
  start_year, end_year, window = 1992, 2012, 3
  lam, omega, seed = 1.0, 1.0, 0

  df = rolling_crime_assignment(
    start_year=start_year,
    end_year=end_year,
    window=window,
    lam=lam,
    omega=omega,
    seed=seed,
    arrest_col='arrest_rate_smooth',
  )

  file_path = _file_path(
    start_year=start_year, end_year=end_year, window=window,
    lam=lam, omega=omega, seed=seed)
  file_path.parents[0].mkdir(parents=True, exist_ok=True)
  df.to_csv(file_path, index=False)


if __name__ == "__main__":
  main()
