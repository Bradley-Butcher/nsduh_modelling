import numpy as np
import pandas as pd

from typing import List
from cj_pipeline.config import logger
from sklearn.linear_model import LinearRegression


def subset_pd_bool(df, **kwargs):
  cond = True
  for key, val in kwargs.items():
    cond &= df[key] == val
  return cond


def subset_pd(df, **kwargs):
  return df[subset_pd_bool(df, **kwargs)]


def smooth_arrest_rates(
    df: pd.DataFrame,
    groups: List[str],
    x_test: np.ndarray,
    x_col: str,
    count_col: str,
    arrest_col: str,
    smooth_col: str,
    mode: str,
) -> pd.DataFrame:
  reg_groups = [g for g in groups if g != x_col]

  def _smooth(group):
    data = group[group[arrest_col].notna()]
    data, smoother = init_smoothing(
      data=data,
      mode=mode,
      arrest_col=arrest_col,
      count_col=count_col
    )
    if len(data) == 0:
      logger.warn(f'no arrest data to smooth for "{arrest_col}" in group: '
                  f'{group[reg_groups].drop_duplicates().iloc[0].to_dict()}')
      return list(zip(x_test.squeeze(1), [None] * len(x_test)))

    x_train = data[x_col].to_numpy()[:, None]
    y_train, weights = data[arrest_col], data[count_col]
    smoothed = smoother(x_train, y_train, weights=weights, x_test=x_test)
    if pd.isna(smoothed).sum() > 0:
      raise RuntimeError('NaN values in smoothed regression')
    return list(zip(x_test.squeeze(1), smoothed))

  smoothed = df.groupby(reg_groups).apply(
    _smooth).to_frame(smooth_col).reset_index()
  smoothed = smoothed.explode(smooth_col)
  smoothed[x_col] = smoothed[smooth_col].str[0].astype(df[x_col].dtype)
  smoothed[smooth_col] = smoothed[smooth_col].str[1]
  return smoothed


def init_smoothing(data, mode, arrest_col, count_col):
  if mode.startswith('lr_'):
    smoother = linear_smoother
  elif mode.startswith('avg_'):
    smoother = avg_smoother
  else:
    raise ValueError(f'Unknown smoother typer "{mode}"')

  if mode.endswith('_pc'):
    data = data[data[count_col] > 0]
  elif mode.endswith('_pr'):
    data = data[data[arrest_col] > 0]
  elif mode.endswith('_all'):
    pass
  else:
    raise ValueError(f'Uknown smoothing mode "{mode}"')

  return data, smoother


def avg_smoother(x_train, y_train, weights, x_test, **_):
  weights /= np.nansum(weights)
  smoothed = [np.nanmean(y_train * weights)] * len(x_test)
  return smoothed


def linear_smoother(x_train, y_train, weights, x_test, eps=0.0):
  model = LinearRegression()
  model.fit(x_train, y_train, weights)
  smoothed = model.predict(x_test).clip(min=eps)
  return smoothed

