import numpy as np
import pandas as pd
from functools import partial
from collections import Counter

from cj_pipeline.config import BASE_DIR, CRIMES, CRIMES_GROUP, NEULAW_TO_NCVS, NEULAW_TO_NSDUH, logger
from cj_pipeline.neulaw.assignment_preprocessing import init_neulaw, init_ncvs, init_nsduh


def get_synth(
    start_year: int, end_year: int, window: int, seed: int = 0,
    lam: float = 1, omega: float = 1,
) -> pd.DataFrame:
  drug_col, arrest_col = 'drugs_any', 'arrest_rate_smooth'
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
      lam=lam, omega=omega, drug_col=drug_col, arrest_col=arrest_col)
    df.to_csv(file_path, index=False)

  return df


def _file_path(start_year, end_year, window, lam, omega, seed):
  file_name = f'lam{lam:.2f}_om{omega:.2f}-{seed}.csv'
  data_path = BASE_DIR / 'data' / 'scratch' / 'synth'
  data_path /= f'{start_year}-{end_year}_{window}'
  return data_path / file_name


def _add_unobserved(pop, lam, arrest_col):
  total_crimes = lam * pop['offense_count'] / pop[arrest_col]
  pop['total_crimes'] = total_crimes.round().astype(pd.Int32Dtype())
  pop['unobserved_crimes'] = pop['total_crimes'] - pop['offense_count']
  pop['unobserved_per_person'] = pop['unobserved_crimes'] / pop['pop_size']
  return pop


def _population_offenses(df, group_all, crimes, lam, arrest_col):
  groups = [col for col in group_all if col != 'offense_category']

  offenses = df.groupby(group_all, as_index=False)['offense_count'].sum()
  offenses = pd.merge(
    offenses,
    df.groupby(groups)['def.uid'].nunique().to_frame('pop_size').reset_index(),
    how='left', on=groups)
  offenses = pd.merge(
    offenses, crimes, how='left', left_on=group_all, right_on=CRIMES_GROUP)
  offenses = offenses.drop(columns=CRIMES_GROUP)  # de-duplicate columns
  offenses = _add_unobserved(offenses, lam=lam, arrest_col=arrest_col)
  return offenses


def _sample_unobserved(df, groups, n_samples_div, rng):
  # sampling function
  def _sample(group):
    n_samples = group['unobserved_crimes'].mean() / n_samples_div
    if group['crime_weight'].sum() <= 0 or n_samples < 1:
      return None  # no crimes of this type (happens for some < 18 categories)
    samples = group['def.uid'].sample(
      n=int(n_samples),
      replace=True,
      weights=group['crime_weight'],
      random_state=rng
    )
    return list(Counter(samples).items())

  # df is melted over crimes -> subset only individual records for `crime`
  grouped = df.groupby(groups)
  samples = grouped.apply(_sample).to_frame('def.uid').reset_index()
  samples = samples[samples['def.uid'].notna()]  # remove categories w/o samples
  samples = samples.explode('def.uid')
  samples['offense_unobserved'] = samples['def.uid'].str[1]
  samples['def.uid'] = samples['def.uid'].str[0]

  return samples


def _window_sampler(
    start_year, end_year, window, lam, omega, arrest_col, drug_col, rng):
  # load data for given time-frame
  neulaw_gen, _ = init_neulaw(start_year, end_year=end_year, melt=True)
  ncvs_gen, _ = init_ncvs(start_year, end_year=end_year)
  nsduh_gen, _ = init_nsduh(start_year, end_year=end_year, drug_col=drug_col)

  def _window(last_year: int, n_samples_div: float = 1.0):
    year = last_year - window
    if not start_year <= year <= end_year:
      raise ValueError(
        f'Last year {last_year} not compatible with start year {start_year} '
        f'and window {window}.')

    # load data for given time-frame
    df = neulaw_gen(year)
    ncvs = ncvs_gen(year)
    nsduh = nsduh_gen(year)
    offense_gen = partial(
      _population_offenses, df=df, lam=lam, arrest_col=arrest_col)

    # merge crime statistics
    offenses_ncvs = offense_gen(group_all=NEULAW_TO_NCVS, crimes=ncvs)
    offenses_nsduh = offense_gen(group_all=NEULAW_TO_NSDUH, crimes=nsduh)

    nsduh_ids = df['offense_category'].isin(['dui', 'drugs'])
    unobs_cols = lambda c: c + ['unobserved_per_person', 'unobserved_crimes']
    df = pd.concat([
      pd.merge(df[~nsduh_ids], offenses_ncvs[unobs_cols(NEULAW_TO_NCVS)],
               how='left', on=NEULAW_TO_NCVS),
      pd.merge(df[nsduh_ids], offenses_nsduh[unobs_cols(NEULAW_TO_NSDUH)],
               how='left', on=NEULAW_TO_NSDUH)
    ])
    df['crime_weight'] = df['unobserved_per_person'] + omega * df['offense_count']

    # sample new unobserved crimes
    _sample = partial(
      _sample_unobserved, df=df, n_samples_div=n_samples_div, rng=rng)
    samples_ncvs = _sample(groups=NEULAW_TO_NCVS)
    samples_nsduh = _sample(groups=NEULAW_TO_NSDUH)

    nsduh_ids = df['offense_category'].isin(('dui', 'drugs'))
    df = pd.concat([
      pd.merge(df[~nsduh_ids], samples_ncvs, how='left', on=NEULAW_TO_NCVS + ['def.uid']),
      pd.merge(df[nsduh_ids], samples_nsduh, how='left', on=NEULAW_TO_NSDUH + ['def.uid'])
    ])
    df['offense_unobserved'] = df['offense_unobserved'].fillna(0).astype('int')
    df['offense_total'] = df['offense_count'] + df['offense_unobserved']

    # convert back into the wide format
    df = pd.pivot_table(
      df, columns='offense_category', values='offense_total', sort=False,
      index=['def.gender', 'def.race', 'def.uid', 'def.dob', 'age_cat']
    ).reset_index()

    return df

  return _window


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
  df = df.groupby(df.columns.difference(CRIMES).to_list()).sum().reset_index()
  return df


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
    drug_col='drugs_any',  # TODO: vary `drug_col`
    arrest_col='arrest_rate_smooth',
  )

  file_path = _file_path(
    start_year=start_year, end_year=end_year, window=window,
    lam=lam, omega=omega, seed=seed)
  file_path.parents[0].mkdir(parents=True, exists_ok=True)
  df.to_csv(file_path, index=False)


if __name__ == "__main__":
  main()
