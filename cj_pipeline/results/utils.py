import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Callable


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
        np.minimum(row[yerr], row[y]), np.minimum(row[yerr], 1 - row[y])
      ])[:, None]
      ax.bar(xpos, row[y], yerr=err, width=width, align='edge', **kwargs)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=tick_rotation)
  grid.map_dataframe(errplot, x=x, y=y, yerr=yerr)

  return grid