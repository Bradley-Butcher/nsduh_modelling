import warnings  # TODO: why subset_pd raises .iteritems() -> .items() warning?
warnings.filterwarnings('ignore')

import pandas as pd
from collections import Counter

from cj_pipeline.config import BASE_DIR, CRIMES_GROUP, NEULAW_TO_NCVS, NEULAW_TO_NSDUH
from cj_pipeline.neulaw.assignment_preprocessing import init_neulaw, init_ncvs, init_nsduh


def subset_pd_bool(df, **kwargs):
  cond = True
  for key, val in kwargs.items():
    cond &= df[key] == val
  return cond


def subset_pd(df, **kwargs):
  return df[subset_pd_bool(df, **kwargs)]


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
    total_crimes = (lam * pop['offense_count'] / pop[arrest_col]).round()
    pop['total_crimes'] = total_crimes.astype(pd.Int32Dtype())
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
  # TODO: values in `unobserved_per_person` seem larger than w/ smoothing!

  # TODO: df is melted, so you need to condition on offense_category
  def _sample_unobserved(groups, offenses):
    ans = []
    for crime in offenses['offense_category'].unique():
      def _sample(group):  # TODO: re-run with different seeds?!
        if (group['offense_category'].unique() != crime).any():
          raise ValueError(f'expected only entries for crime "{crime}"; '
                           f'got {group["offense_category"].unique()}')
        if group['crime_weight'].sum() <= 0.0:
          return None  # no crimes of this type (happens for some < 18 categories)
        n_samples = group[group['offense_category'] == crime]['unobserved_crimes']
        n_samples = int(n_samples.mean())  # all are the same
        samples = group.sample(
          n=n_samples, replace=True, weights=group['crime_weight'])
        return list(Counter(samples['def.uid']).items())

      # df is melted over crimes -> subset only individual records for `crime`
      grouped = subset_pd(df, offense_category=crime).groupby(groups)
      samples = grouped.apply(_sample).to_frame('def.uid').reset_index()
      samples = samples[samples['def.uid'].notna()]  # remove categories w/o samples
      samples = samples.explode('def.uid')
      samples['offense_unobserved'] = samples['def.uid'].str[1]
      samples['def.uid'] = samples['def.uid'].str[0]

      ans.append(samples)

    return pd.concat(ans)

  samples_ncvs = _sample_unobserved(NEULAW_TO_NCVS, offenses=offenses_ncvs)
  samples_nsduh = _sample_unobserved(NEULAW_TO_NSDUH, offenses=offenses_nsduh)

  nsduh_ids = df['offense_category'].isin(('dui', 'drugs'))
  df = pd.concat([
    pd.merge(df[~nsduh_ids], samples_ncvs, how='left', on=NEULAW_TO_NCVS + ['def.uid']),
    pd.merge(df[nsduh_ids], samples_nsduh, how='left', on=NEULAW_TO_NSDUH + ['def.uid'])
  ])
  df['offense_unobserved'] = df['offense_unobserved'].fillna(0).astype('int')
  df['offense_total'] = df['offense_count'] + df['offense_unobserved']

  df = pd.pivot_table(
    df, columns='offense_category', values='offense_total', sort=False,
    index=[
      'def.gender', 'def.race', 'def.uid', 'def.dob', 'year_range'
      # 'age', 'age_ncvs', 'age_nsduh',
    ]
  ).reset_index()

  return df


if __name__ == "__main__":
  start_year, window = 1992, 20   # 2009, 3
  data_path = BASE_DIR / 'data' / 'processed'
  df = crime_assignment(start_year=start_year, window=window)
  df.to_csv(data_path / f'synth_crimes.csv', index=False)
  # df.to_csv(data_path / f'synth_{start_year}_{window}.csv', index=False)
