import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cj_pipeline.results.compare_ates import aggregate
from cj_pipeline.counterfactual_matching import average_treatment_effect

from typing import Callable, Tuple


def _nan2none(val):
  return None if pd.isna(val) else val


def barplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    yerr: str,
    hue: str,
    width: float,
    gap: float,
    tick_rotation: int = 0,
    xlabel_map: Callable[[float], str] = None,
    err_clip: Tuple[float, float] = (-np.inf, np.inf),
    **kwargs,
):
  xs, hues = df[x].unique(), df[hue].unique()
  group_width = len(hues) * width
  positions = {
    _nan2none(x_i): {
      _nan2none(h_i): i_x * (group_width + gap) + i_h * width
      for i_h, h_i in enumerate(hues)  # within bar group
    }
    for i_x, x_i in enumerate(xs)  # different bar groups
  }
  xticks, xlabels = list(zip(*[
    (i_x * (group_width + gap) + group_width / 2, x_i)
    for i_x, x_i in enumerate(xs)
  ]))
  if xlabel_map is not None:
    xlabels = list(map(xlabel_map, xlabels))

  grid = sns.FacetGrid(
    df,
    hue=hue,
    margin_titles=True,
    despine=True,
    **kwargs
  )

  def errplot(x, y, yerr, **kwargs):
    ax = plt.gca()
    data = kwargs.pop('data')
    for _, row in data.iterrows():
      xpos = positions[_nan2none(row[x])][_nan2none(row[hue])]
      err = np.array([
        np.minimum(row[yerr], row[y] - err_clip[0]),
        np.minimum(row[yerr], err_clip[1] - row[y]),
      ])[:, None]
      ax.bar(xpos, row[y], yerr=err, width=width, align='edge', **kwargs)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=tick_rotation)
  grid.map_dataframe(errplot, x=x, y=y, yerr=yerr)

  return grid


def load_observed():
  # CAVEAT: assumes synth is based on run of `aggregate` & that binning didn't change
  synth = aggregate(drop_constant_cols=False)
  exp = synth[synth.columns[synth.nunique() == 1]]
  exp = exp.drop_duplicates().iloc[0]  # only one row by def

  # ignoring git commit information for now
  ates, cates = average_treatment_effect(
    start_year=exp.start_year,
    end_year=exp.end_year,
    treatment='calc.race',
    binary_treatment_set={exp.baseline: 0, exp.treatment: 1},
    use_synth=False,  # ensures this is based on non-synthetic data
    matching_alg=exp.matching,
    repeat_match=exp.repeat_match,
    n_subsample=exp.n_subsample,
    crime_bins=tuple(int(n) for n in exp.crime_bins.split()),
    seed=0,  # TODO: only affects subsampling -> don't iterate atm
    smoothing=exp.smoothing,
    rate_mult_ncvs=None, rate_mult_nsduh=None,  # does not affect non-synth
  )

  return ates, cates