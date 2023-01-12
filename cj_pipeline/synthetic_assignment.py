import warnings  # TODO: why subset_pd raises .iteritems() -> .items() warning?
warnings.filterwarnings('ignore')

import pandas as pd
from collections import Counter

from cj_pipeline.config import CRIMES_GROUP, NEULAW_GROUP, NEULAW_TO_NCVS, NEULAW_TO_NSDUH
from cj_pipeline.neulaw.assignment_preprocessing import init_neulaw, init_ncvs, init_nsduh


def subset_pd_bool(df, **kwargs):
  cond = True
  for key, val in kwargs.items():
    cond &= df[key] == val
  return cond


# def population_stats_preprocessing(population: pd.DataFrame) -> pd.DataFrame:
#   # TODO: necessary pre-processing may vary for different year windows
#
#   # set zero sex offense arrest rate to mean reporting rate
#   row_cond = {'def.gender': 'Female', 'offense_category': 'sex offense'}
#   row_cond = subset_pd_bool(population, **row_cond)
#   population.loc[row_cond, 'arrest_rate'] = population[row_cond]['reporting_rate'].mean()
#
#   # set zero dui arrest rate to the next closest category
#   row_cond = {'def.gender': 'Female', 'def.race': 'Black', 'offense_category': 'dui'}
#   zero_idx = subset_pd_bool(population, age_cat='< 18', **row_cond)
#   adj_idx = subset_pd_bool(population, age_cat='18-30', **row_cond)
#   population.loc[zero_idx, 'arrest_rate'] = population[adj_idx]['arrest_rate'].mean()
#
#   return population


def crime_assignment(
    start_year: int, window: int, lam: float = 1.0, omega: float = 1.0,
    drug_col: str = 'drugs_any', arrest_col: str = 'arrest_rate_smooth'
    # TODO: vary `drug_col`
):
  # load data for given time-frame
  neulaw_gen, _ = init_neulaw(start_year, window=window)
  ncvs_gen, _ = init_ncvs(start_year, window=window)
  nsduh_gen, _ = init_nsduh(start_year, window=window, drug_col=drug_col)

  df = neulaw_gen(start_year)
  ncvs = ncvs_gen(start_year)
  nsduh = nsduh_gen(start_year)

  def _add_unobserved(pop):
    pop['total_crimes'] = lam * pop['offense_count'] / pop[arrest_col]
    pop['unobserved_crimes'] = pop['total_crimes'] - pop['offense_count']
    pop['unobserved_per_person'] = pop['unobserved_crimes'] / pop['pop_size']
    return pop

  def _offenses(group_all, crimes):
    groups = [col for col in group_all if col != 'offense_category']
    offenses = df.groupby(group_all, as_index=False)['offense_count'].sum()
    offenses = pd.merge(
      offenses,
      df.groupby(groups)['def.uid'].nunique().to_frame('pop_size').reset_index(),
      how='left', on=groups)
    offenses = pd.merge(
      offenses, crimes, how='left', left_on=group_all, right_on=CRIMES_GROUP)
    offenses = offenses.drop(columns=CRIMES_GROUP)  # de-duplicate columns
    # TODO: pre-processing necessary w/o smoothing?
    offenses = _add_unobserved(offenses)
    return offenses

  # merge crime statistics
  offenses_ncvs = _offenses(NEULAW_TO_NCVS, crimes=ncvs)
  offenses_nsduh = _offenses(NEULAW_TO_NSDUH, crimes=nsduh)

  # merge the counts into the original df
  nsduh_ids = df['offense_category'].isin(('dui', 'drugs'))
  unobs_cols = lambda c: c + ['unobserved_per_person', 'unobserved_crimes']
  df = pd.concat([
    pd.merge(df[~nsduh_ids], offenses_ncvs[unobs_cols(NEULAW_TO_NCVS)],
             how='left', on=NEULAW_TO_NCVS),
    pd.merge(df[nsduh_ids], offenses_nsduh[unobs_cols(NEULAW_TO_NSDUH)],
             how='left', on=NEULAW_TO_NSDUH)
  ])
  df['crime_weight'] = df['unobserved_per_person'] + omega * df['offense_count']

  # TODO: the values in `unobserved_per_person` seem larger than w/o smoothing!!!

  # sampling code  # TODO: re-run with different seeds?!
  def _sample(group):
    if group['crime_weight'].sum() <= 0.0:
      return None  # no crimes of this type (happens for some < 18 categories
    samples = group.sample(
      n=int(group['unobserved_crimes'].mean()),
      replace=True, weights=group['crime_weight'])
    return list(Counter(samples['def.uid']).items())

  groups = NEULAW_GROUP + ['age_cat']
  samples = df.groupby(groups).apply(_sample).to_frame('def.uid').reset_index()
  samples = samples[samples['def.uid'].notna()]  # remove categories w/o samples
  samples = samples.explode('def.uid')
  samples['offense_unobserved'] = samples['def.uid'].str[1]
  samples['def.uid'] = samples['def.uid'].str[0]

  df = pd.merge(df, samples, how='left', on=groups + ['def.uid'])
  df['offense_unobserved'] = df['offense_unobserved'].fillna(0).astype('int')

  return df


if __name__ == "__main__":
  crime_assignment(start_year=2009, window=3)
