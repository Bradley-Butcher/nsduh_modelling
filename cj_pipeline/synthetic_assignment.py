# We have criminal history - specifically arrests/convictions
# We have some number of synthetic crimes to assign
# Must be conditioned on crime type

# Find a period of time (N year window, 3 by default)
# For each year, if person was 18 or above, how many offenses of each type?

# Assign unobserved offenses, based on their criminal activity

import os
import warnings  # TODO: why subset_pd raises .iteritems() -> .items() warning?
warnings.filterwarnings('ignore')

import pandas as pd
from collections import Counter
from cj_pipeline.neulaw.assignment_preprocessing import init_offence_counting


NEULAW_GROUP = ['def.gender', 'def.race', 'age_cat', 'offense_category']
CRIMES_GROUP = ['offender_sex', 'offender_race', 'offender_age', 'crime_recode']


def subset_pd_bool(df, **kwargs):
  cond = True
  for key, val in kwargs.items():
    cond &= df[key] == val
  return cond


def neulaw_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
  # handle missing and special values
  df = df[df['def.gender'].isin(('Female', 'Male'))]
  df = df[~df['def.race'].isin(('Unknown', 'Missing'))]
  df = df.replace({'def.race': {'Asian': 'Other', 'Indian': 'Other'}})

  # convert from wide to tall
  categories = ['aggravated assault', 'drugs', 'dui', 'property',
                'robbery', 'sex offense', 'simple assault']
  df = df.melt(
    id_vars=df.columns.difference(categories), value_vars=categories,
    var_name='offense_category', value_name='offense_count')

  # validate the dataframe
  n_duplicates = df.groupby(NEULAW_GROUP).apply(
    lambda g: len(g) != g['def.uid'].nunique()).sum()
  if n_duplicates > 0:  # some group has multiple entries per individual
    raise RuntimeError('Duplicated entries after `pd.melt` of NEULAW.')

  return df


def neulaw_population_stats(df: pd.DataFrame) -> pd.DataFrame:
  # count arrests in population
  population = df.groupby(NEULAW_GROUP, as_index=False)['offense_count'].sum()

  # add population sizes
  groups = [col for col in NEULAW_GROUP if col != 'offense_category']
  population = pd.merge(  # add population sizes
    population, df.groupby(groups, as_index=False)['def.uid'].nunique(),
    how='left', on=groups)
  population = population.rename(columns={'def.uid': 'population_size'})

  return population


def ncvs_nsduh_crime_stats(data_path: str, drug_col: str) -> pd.DataFrame:
  dui_col = 'dui_arrest_rate'
  ncvs = pd.read_csv(os.path.join(data_path, 'ncvs.csv'))  # TODO: add window param
  nsduh = pd.read_csv(os.path.join(data_path, 'nsduh.csv'))  # TODO: add window param

  crimes = nsduh.melt(
    id_vars=[col for col in CRIMES_GROUP if col != 'crime_recode'],
    value_vars=[dui_col, drug_col],
    var_name='crime_recode', value_name='arrest_rate')
  crimes = crimes.replace({'crime_recode': {dui_col: 'dui', drug_col: 'drugs'}})
  crimes = pd.concat([ncvs, crimes])  # fills non-matching columns with NaN

  return crimes


def population_stats_preprocessing(population: pd.DataFrame) -> pd.DataFrame:
  # TODO: necessary pre-processing may vary for different year windows

  # set zero sex offense arrest rate to mean reporting rate
  row_cond = {'def.gender': 'Female', 'offense_category': 'sex offense'}
  row_cond = subset_pd_bool(population, **row_cond)
  population.loc[row_cond, 'arrest_rate'] = population[row_cond]['reporting_rate'].mean()

  # set zero dui arrest rate to the next closest category
  row_cond = {'def.gender': 'Female', 'def.race': 'Black', 'offense_category': 'dui'}
  zero_idx = subset_pd_bool(population, age_cat='< 18', **row_cond)
  adj_idx = subset_pd_bool(population, age_cat='18-30', **row_cond)
  population.loc[zero_idx, 'arrest_rate'] = population[adj_idx]['arrest_rate'].mean()

  return population


def crime_assignment(
    data_path: str, start_year: int, window: int,  # TODO: vary `drug_col`
    lam: float = 1.0, omega: float = 1.0, drug_col: str = 'drugs_any_arrest_rate'):
  offense_counts, max_year = init_offence_counting(start_year=start_year, window=window)

  # neulaw arrest statistics for given time-frame
  df = offense_counts(start_year)
  df = neulaw_preprocessing(df)
  population = neulaw_population_stats(df)

  # population crime statistics
  crimes = ncvs_nsduh_crime_stats(data_path=data_path, drug_col=drug_col)

  # merge data
  population = pd.merge(
    population, crimes, how='left', left_on=NEULAW_GROUP, right_on=CRIMES_GROUP)
  population = population.drop(columns=CRIMES_GROUP)  # de-duplicate columns

  # treat missing and zero values
  population = population_stats_preprocessing(population)

  # compute naive redistribution
  population['total_crimes'] = lam * population['offense_count'] / population['arrest_rate']
  population['unobserved_crimes'] = population['total_crimes'] - population['offense_count']
  population['unobserved_per_person'] = population['unobserved_crimes'] / population['population_size']

  # merge in the counts into the original
  df = pd.merge(
    df, population[NEULAW_GROUP + ['unobserved_per_person', 'unobserved_crimes']],
    how='left', on=NEULAW_GROUP)
  df['crime_weight'] = df['unobserved_per_person'] + omega * df['offense_count']

  # sampling code  # TODO: re-run with different seeds?!
  def _sample(group):
    if group['crime_weight'].sum() <= 0.0:
      return None  # no crimes of this type (happens for some < 18 categories

    samples = group.sample(
      n=int(group['unobserved_crimes'].mean()),
      replace=True,
      weights=group['crime_weight']
    )

    return list(Counter(samples['def.uid']).items())
    # return list(samples['def.uid'])

  samples = df.groupby(NEULAW_GROUP).apply(_sample).to_frame('def.uid').reset_index()
  samples = samples[samples['def.uid'].notna()]  # remove categories w/o samples
  samples = samples.explode('def.uid')
  samples['offense_unobserved'] = samples['def.uid'].str[1]
  samples['def.uid'] = samples['def.uid'].str[0]

  df = pd.merge(df, samples, how='left', on=NEULAW_GROUP + ['def.uid'])
  df['offense_unobserved'] = df['offense_unobserved'].fillna(0).astype('int')

  return df  # TODO: decide what to return / refactor


if __name__ == "__main__":
  data_path = os.path.join(os.getcwd(), '..', 'data', 'processed')
  crime_assignment(data_path=data_path, start_year=2000, window=3)

