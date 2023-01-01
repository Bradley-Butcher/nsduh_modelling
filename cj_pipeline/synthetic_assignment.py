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


def subset_pd_bool(df, **kwargs):
  cond = True
  for key, val in kwargs.items():
    cond &= df[key] == val
  return cond


def crime_assignment(start_year: int, window: int):
  lam, omega, start_year, window = 1.0, 1.0, 2000, 3   # TODO: delete

  ncvs = pd.read_csv('data/processed/ncvs.csv')  # TODO: add window param
  nsduh = pd.read_csv('data/processed/nsduh.csv')  # TODO: add window param
  offense_counts = init_offence_counting(start_year=start_year, window=window)


  ## ncvs formula
  df = offense_counts(2000)  # TODO: delete

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

  # count arrests in population
  pop_groups = ['def.gender', 'def.race', 'age_cat', 'offense_category']  # TODO: unify var naming
  population = df.groupby(pop_groups, as_index=False)['offense_count'].sum()

  # add population sizes
  groups = [col for col in pop_groups if col != 'offense_category']
  population = pd.merge(  # add population sizes
    population, df.groupby(groups, as_index=False)['def.uid'].nunique(),
    how='left', on=groups)
  population = population.rename(columns={'def.uid': 'population_size'})

  # merge data
  ncvs_groups = ['offender_sex', 'offender_race', 'offender_age', 'crime_recode']
  population = pd.merge(
    population, ncvs, how='left', left_on=pop_groups, right_on=ncvs_groups)
  population = population.drop(columns=ncvs_groups)  # de-duplicate columns
  # TODO: merge in the nsduh data

  # preprocess arrest_rates where zero (differs between population and ncvs)
  row_cond = {'def.gender': 'Female', 'offense_category': 'sex offense'}
  row_cond = subset_pd_bool(population, **row_cond)
  population.loc[row_cond, 'arrest_rate'] = population[row_cond]['reporting_rate'].mean()
  ## set zero arrest rate to mean reporting rate

  # compute naive redistribution
  population['naive_count'] = lam * population['offense_count'] / population['arrest_rate']
  population['naive_count'] -= population['offense_count']  # -> total unobserved crimes
  population['naive_count'] /= population['population_size']

  # merge in the counts into the original
  df = pd.merge(
    df, population[pop_groups + ['naive_count']], how='left', on=pop_groups)
  df['crime_weight'] = df['naive_count'] + omega * df['offense_count']
