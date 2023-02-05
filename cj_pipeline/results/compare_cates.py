import json
import pathlib
import numpy as np
import pandas as pd

from cj_pipeline.results import utils
from cj_pipeline.config import BASE_DIR

DEFAULT_DIR = BASE_DIR / pathlib.Path('cj_pipeline/results/data')


def _load_cate(path_object):
  # with open(path_object, 'r') as f:
  #   experiment = json.load(f)
  #   experiment['crime_bins'] = ' '.join(experiment['crime_bins'])
  fname = path_object.name[:path_object.name.rfind('.')]
  cate = pd.read_csv(path_object.parents[0] / f"{fname}-cate.csv")
  return cate


def _load_data(data_path):
  data_path = pathlib.Path(data_path)
  if not data_path.is_absolute():
    data_path = BASE_DIR / data_path

  # load observed data
  observed_paths = list(data_path.rglob('*_observed_*.json'))
  if len(observed_paths) == 0:
    _, observed = utils.load_observed()
  elif len(observed_paths) == 1:
    observed = _load_cate(observed_paths[0])
  else:
    raise ValueError(f'{len(observed_paths)} non-synth results in "{data_path}"')

  # load synth data
  synth = []
  for path_object in data_path.rglob('*.json'):
    if '_observed_' in path_object.name:
      continue
    synth.append(_load_cate(path_object))
  synth = pd.concat(synth)

  return observed, synth


def aggregate(
    data_path: str | pathlib.Path = DEFAULT_DIR,
    remove_mixed: bool = False,
) -> pd.DataFrame:
  val_col, weight_col = 'cate', 'group_size'
  observed, synth = _load_data(data_path)

  def _mean(group):
    mean = np.average(group[val_col], weights=group[weight_col])
    return mean

  def _sem(group):
    mean = _mean(group)
    variance = np.average((group[val_col] - mean)**2, weights=group[weight_col])
    sem = (variance / len(group))**0.5
    return sem

  groups = ['score', 'age_cat', 'def.gender']
  means = pd.merge(
    observed.groupby(groups).apply(_mean).to_frame('observed').reset_index(),
    synth.groupby(groups).apply(_mean).to_frame('synth').reset_index(),
    on=groups, how='outer',
  )
  sems = pd.merge(
    observed.groupby(groups).apply(_sem).to_frame('observed_sem').reset_index(),
    synth.groupby(groups).apply(_sem).to_frame('synth_sem').reset_index(),
    on=groups, how='outer',
  )
  results = pd.merge(means, sems, how='inner', on=groups)
  results['diff'] = results['synth'] - results['observed']
  if remove_mixed:
    for col in groups:
      results = results[results[col] != '*']

  return results


if __name__ == '__main__':
  data_path = DEFAULT_DIR
  results = aggregate(data_path, remove_mixed=True)
  results.to_csv(data_path / 'results_cate.csv', index=False)
