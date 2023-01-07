# We have criminal history - specifically arrests/convictions
# We have some number of synthetic crimes to assign
# Must be conditioned on crime type

# Find a period of time (N year window, 3 by default)
# For each year, if person was 18 or above, how many offenses of each type?

# Assign unobserved offenses, based on their criminal activity


import pandas as pd
from cj_pipeline.neulaw.assignment_preprocessing import init_offence_counting

import warnings  # TODO: why subset_pd raises .iteritems() -> .items() warning?
warnings.filterwarnings('ignore')


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


def ncvs_nsduh_crime_stats(drug_col: str) -> pd.DataFrame:
  dui_col = 'dui_arrest_rate'
  ncvs = pd.read_csv('data/processed/ncvs.csv')  # TODO: add window param
  nsduh = pd.read_csv('data/processed/nsduh.csv')  # TODO: add window param

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
    start_year: int, window: int, lam: float = 1.0, omega: float = 1.0):
  # lam, omega, start_year, window = 1.0, 1.0, 2000, 3
  offense_counts, max_year = init_offence_counting(start_year=start_year, window=window)

  # neulaw arrest statistics for given time-frame
  df = offense_counts(start_year)
  df = neulaw_preprocessing(df)
  population = neulaw_population_stats(df)

  # population crime statistics  # TODO: play with different `drug_col` choices
  crimes = ncvs_nsduh_crime_stats(drug_col='drugs_any_arrest_rate')

  # merge data
  population = pd.merge(
    population, crimes, how='left', left_on=NEULAW_GROUP, right_on=CRIMES_GROUP)
  population = population.drop(columns=CRIMES_GROUP)  # de-duplicate columns

  # treat missing and zero values
  population = population_stats_preprocessing(population)

  # compute naive redistribution
  population['naive_count'] = population['offense_count'] / population['arrest_rate']
  population['naive_count'] -= population['offense_count']  # -> total unobserved crimes
  population['naive_count'] *= lam / population['population_size']

  # merge in the counts into the original
  df = pd.merge(
    df, population[NEULAW_GROUP + ['naive_count']], how='left', on=NEULAW_GROUP)
  df['crime_weight'] = df['naive_count'] + omega * df['offense_count']


