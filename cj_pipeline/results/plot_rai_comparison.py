import pandas as pd
import seaborn as sns

from cj_pipeline.results import utils
from cj_pipeline.config import SCORES, BASE_DIR
from cj_pipeline.results.compare_ates import aggregate


def _load_data(use_offset, ignore_cols=None):
  synth = aggregate(drop_constant_cols=True, ignore_cols=ignore_cols)
  if 'synth' in synth.columns and not all(synth['synth']):
    observed = synth[~synth['synth']]
    synth = synth[synth['synth']]
  else:
    synth['synth'] = True
    observed = _load_observed() if use_offset else pd.DataFrame(columns=synth.columns)
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

  id_vars = df.columns[df.nunique() > 1].union({'synth'})
  id_vars = [c for c in id_vars if not c[:c.find('_')] in SCORES]
  df = pd.merge(
    _melt(id_vars, suffix='_mean', value_name='mean'),
    _melt(id_vars, suffix='_sem', value_name='sem'),
  )
  return df


def _load_observed():
  ate, _ = utils.load_observed()
  observed = pd.DataFrame({
    f'{s}_mean': [ate[ate.score == s]['ate'].iloc[0]] for s in SCORES
  })
  observed['synth'] = False
  return observed


def plot_rais(df, gap=0.1, width=0.2, use_offset=False, exclude=None):
  sns.set_style('whitegrid')
  sns.set(font_scale=1.25)

  synth, observed = df[df.synth], df[~df.synth] if use_offset else None
  for score in SCORES:
    idx = synth.score == score
    offset = observed[observed.score == score]['mean'].iloc[0] if use_offset else 0
    synth.loc[idx, 'mean'] = 100 * (synth[idx]['mean'] - offset)
    synth.loc[idx, 'sem'] = 100 * synth[idx]['sem']

  if exclude is not None:
    synth = synth[~synth.score.isin(exclude)]

  synth['ci'] = 1.96 * synth['sem']
  synth['score'] = synth['score'].str.upper()
  xlabel_map = lambda lbl: 'est' if pd.isna(lbl) else str(lbl)
  grid = utils.barplot(
    df=synth,
    x='lam', y='mean', yerr='ci',
    hue='omega', col='score',
    width=width, gap=gap,
    xlabel_map=xlabel_map,
    err_clip=(-100.0, 100.0) if use_offset else (0.0, 100.0),
  )

  grid.fig.supxlabel(r'$\lambda$')
  grid.set_axis_labels(
    x_var='', y_var='% point score change' if use_offset else 'Estimated effect')
  grid.set_titles(row_template='{row_name}', col_template='{col_name}')
  grid.add_legend(title=r'$\omega$')

  grid.figure.savefig(
    BASE_DIR / 'data' / 'scratch' / f'rais_{"offset" if use_offset else "raw"}.pdf')


if __name__ == '__main__':
  use_offset = True
  ignore_cols = ['commit']
  df = _load_data(use_offset=use_offset, ignore_cols=ignore_cols)
  plot_rais(df, exclude=['fta'], use_offset=use_offset)