import pandas as pd
import seaborn as sns

from cj_pipeline.config import SCORES, BASE_DIR
from cj_pipeline.results.utils import barplot
from cj_pipeline.results.compare_ates import aggregate
from cj_pipeline.counterfactual_matching import average_treatment_effect


def _load_data(ignore_cols=None):
  synth = aggregate(drop_constant_cols=True, ignore_cols=ignore_cols)
  if 'synth' in synth.columns and not all(synth['synth']):
    observed = synth[~synth['synth']]
    synth = synth[synth['synth']]
  else:
    synth['synth'] = True
    observed = _load_observed()
  df = pd.concat([observed, synth], ignore_index=True)

  def _melt(id_vars, suffix, value_name):
    ans = df.melt(
      id_vars=id_vars,
      value_vars=[c for c in df.columns if c.endswith(suffix)],
      var_name='score', value_name=value_name,
    )
    ans['score'].replace(
      {v: v[:v.rfind('_')] for v in ans['score'].unique()},
      inplace=True,
    )
    return ans

  id_vars = df.columns[df.nunique() > 1]
  id_vars = [c for c in id_vars if not c[:c.find('_')] in SCORES]
  # id_vars = ['lam', 'omega', 'synth']
  df = pd.merge(
    _melt(id_vars, suffix='_mean', value_name='mean'),
    _melt(id_vars, suffix='_sem', value_name='sem'),
  )
  return df


def _load_observed():
  # CAVEAT: assumes synth is based on run of `aggregate` & that binning didn't change
  synth = aggregate(drop_constant_cols=False)
  exp = synth[synth.columns[synth.nunique() == 1]]
  exp = exp.drop_duplicates().iloc[0]  # only one row by def

  # ignoring git commit information for now
  ate, _ = average_treatment_effect(
    start_year=exp.start_year,
    end_year=exp.end_year,
    treatment='calc.race',
    binary_treatment_set={exp.baseline: 0, exp.treatment: 1},
    use_synth=False,
    matching_alg=exp.matching,
    repeat_match=exp.repeat_match,
    n_subsample=exp.n_subsample,
    crime_bins=tuple(int(n) for n in exp.crime_bins.split()),
    seed=0,  # TODO: only affects subsampling -> don't iterate atm
    smoothing=exp.smoothing,
    rate_mult_ncvs=None, rate_mult_nsduh=None,  # does not affect non-synth
  )

  observed = pd.DataFrame({
    f'{s}_mean': [ate[ate.score == s]['ate'].iloc[0]] for s in SCORES
  })
  observed['synth'] = False
  return observed


def plot_rais(df, gap=0.1, width=0.2, exclude=None):
  sns.set_style('whitegrid')
  sns.set(font_scale=1.25)

  synth, observed = df[df.synth], df[~df.synth]
  for score in SCORES:
    idx = synth.score == score
    offset = observed[observed.score == score]['mean'].iloc[0]
    synth.loc[idx, 'mean'] = 100 * (synth[idx]['mean'] - offset)
    synth.loc[idx, 'sem'] = 100 * synth[idx]['sem']

  if exclude is not None:
    synth = synth[~synth.score.isin(exclude)]

  synth['ci'] = 1.96 * synth['sem']
  synth['score'] = synth['score'].str.upper()
  xlabel_map = lambda lbl: 'est' if pd.isna(lbl) else str(lbl)
  grid = barplot(
    df=synth,
    x='lam', y='mean', yerr='ci',
    hue='omega', col='score',
    width=width, gap=gap,
    xlabel_map=xlabel_map,
    err_clip=(-100.0, 100.0),
  )

  grid.fig.supxlabel(r'$\lambda$')
  grid.set_axis_labels(x_var='', y_var='% point score change')
  grid.set_titles(row_template='{row_name}', col_template='{col_name}')
  grid.add_legend(title=r'$\omega$')

  grid.figure.savefig(BASE_DIR / 'data' / 'scratch' / 'rais.pdf')


if __name__ == '__main__':
  ignore_cols = ['commit', 'crime_bins']
  df = _load_data(ignore_cols=ignore_cols)
  plot_rais(df, exclude=['fta'])