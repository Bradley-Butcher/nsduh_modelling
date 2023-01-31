import git
import json
import datetime
import numpy as np
import pandas as pd
from absl import app, flags

from cj_pipeline.neulaw.assignment_preprocessing import init_neulaw
from cj_pipeline.neulaw.preprocess import init_rai_year_range
from cj_pipeline.config import logger, BASE_DIR, CRIMES, DEMOGRAPHICS
from cj_pipeline.synthetic_assignment import get_synth

import dame_flame

SCORES = [
  'nca',
  'nvca',
  'ogrs3',
  'vprai',
  'fta'
]
MATCHING_ALGS = [
  'dame',
  'flame',
  'hybrid',
]
ETHNICITIES = [
  'Black',
  'White',
  'Hispanic',
]

FLAGS = flags.FLAGS
flags.DEFINE_integer('start_year', 1992, 'Initial year of records')
flags.DEFINE_integer('end_year', 2012, 'Initial year of records')
flags.DEFINE_integer('n_subsample', None, 'Number of subsamples')
flags.DEFINE_bool('repeat_match', False, 'Repeat samples in matching')
flags.DEFINE_enum(
  'matching', 'flame', MATCHING_ALGS, help=f'One of {MATCHING_ALGS}.')
flags.DEFINE_enum(
  'baseline', 'White', ETHNICITIES, help=f'One of {ETHNICITIES}')
flags.DEFINE_enum(
  'treatment', 'Black', ETHNICITIES, help=f'One of {ETHNICITIES}')
flags.DEFINE_spaceseplist(
  'crime_bins', "-1 0 1 2 4 9 100000",
  'Right ends of the crime bins (inclusive); e.g., "[-1, 1, 9, 1000]".')

flags.DEFINE_bool('synth', False, 'Use synthetic data.')
flags.DEFINE_integer('window', 2, 'No. of years to prepend.')
flags.DEFINE_float('lam', None, 'Multiplier of total crimes.')
flags.DEFINE_float('omega', 1.0, 'Multiplier of recorded crimes.')
flags.DEFINE_integer('seed', 0, 'Seed for the random sample generation.')


def _min_max_scale(df: pd.DataFrame) -> pd.DataFrame:
  return (df - df.min()) / (df.max() - df.min())


def _binarize_treatment(
    df: pd.DataFrame,
    treatment: str,
    binary_treatment_set: dict[str, int]
):
  if len(binary_treatment_set) != 2:
    raise ValueError("binary_treatment_set must have 2 keys")

  df = df.copy()
  if "Rest" in binary_treatment_set.keys():
    values = df[treatment].unique()
    other = set(values) - set(binary_treatment_set.keys())
    for o_i in other:
      binary_treatment_set[o_i] = binary_treatment_set["Rest"]
  df = df[df[treatment].isin(list(binary_treatment_set.keys()))]
  df[treatment] = df[treatment].map(binary_treatment_set)
  return df, {'treated': {v: k for k, v in binary_treatment_set.items()}}


def _binarize_crimes(df: pd.DataFrame, bins: list):
  codes = {}
  _proc = lambda i: str(i.right) if i.length == 1 else f'{i.left + 1}-{i.right}'
  for crime in CRIMES:
    categorical = pd.Categorical(pd.cut(df[crime], right=True, bins=bins))
    codes[crime] = dict(enumerate(map(_proc, categorical.categories)))
    df[crime] = categorical.codes
  return df, codes


def _binarize_demographics(df, treatment):
  codes = {}
  for dem in set(DEMOGRAPHICS) - {treatment}:
    categorical = pd.Categorical(df[dem])
    codes[dem] = dict(enumerate(categorical.categories))
    df[dem] = categorical.codes
  return df, codes


def _matching_model(score_df, matching_alg, repeat_match):
  matching_alg = matching_alg.lower()
  if matching_alg not in MATCHING_ALGS:
    raise ValueError(f'Unknown matching algorithm "{matching_alg}"')

  if matching_alg == 'dame':
    model = dame_flame.matching.DAME(repeats=repeat_match)
  else:
    model = dame_flame.matching.FLAME(repeats=repeat_match)
  kwargs = {'pre_dame': 1} if matching_alg == 'hybrid' else {}

  model.fit(score_df)
  cates = model.predict(score_df, **kwargs).drop_duplicates()
  cates['cate'] = dame_flame.utils.post_processing.CATE(model, cates.index)

  return model, cates


def average_treatment_effect(
    start_year: int,
    end_year: int,
    treatment: str,
    binary_treatment_set: dict[str, int],
    use_synth: bool = False,
    matching_alg: str = 'flame',
    repeat_match: bool = False,
    n_subsample: int = None,
    seed: int = None,
    crime_bins: tuple = (-1, 0, 1, 2, 9, 100_000),
    **kwargs  # passed to the synthetic assignment when `use_synth` is true
):
  rng = np.random.RandomState(seed)

  logger.info('Loading RAI scores')
  rai_func = init_rai_year_range(start_year=start_year, end_year=end_year)
  rai_dfs = rai_func(start_year)

  logger.info('Loading offense counts')
  if use_synth:
    logger.info('Estimating based on synthethic crime record')
    offense_dfs = get_synth(
      start_year=start_year, end_year=end_year, seed=seed, **kwargs)
  else:
    logger.info('Estimating based on recorded crime only')
    offense_count_func, _ = init_neulaw(
      start_year=start_year, window=end_year - start_year)
    offense_dfs = offense_count_func(start_year)

  codes = {}
  df = pd.merge(offense_dfs, rai_dfs, on=['def.uid'])
  df = df[SCORES + CRIMES + DEMOGRAPHICS]
  df, crime_codes = _binarize_crimes(df, bins=list(crime_bins))
  df, treat_codes = _binarize_treatment(df, treatment, binary_treatment_set)
  df, dem_codes = _binarize_demographics(df, treatment=treatment)
  codes.update(crime_codes); codes.update(treat_codes); codes.update(dem_codes)

  all_score_df = _min_max_scale(df[SCORES])
  df = df.drop(columns=SCORES)
  df = df.rename(columns={treatment: 'treated'})

  results = []
  conditional_results = []
  for score in SCORES:
    logger.info(f"Calculating ATE for treatment: {treatment}. outcome: {score}")
    score_df = df.copy()

    score_df["outcome"] = all_score_df[score]
    before_len = len(score_df)
    score_df = score_df.dropna(subset=["outcome"])
    n_dropped = before_len - len(score_df)
    logger.info(
      f"Dropped {n_dropped} rows with missing values for {score}")

    if n_subsample is not None:
      n_subsample = min(n_subsample, len(score_df))
      score_df = score_df.sample(n=n_subsample, random_state=rng)

    model, cates = _matching_model(
      score_df, matching_alg=matching_alg, repeat_match=repeat_match)
    ate = dame_flame.utils.post_processing.ATE(matching_object=model)
    att = dame_flame.utils.post_processing.ATT(matching_object=model)
    cates = cates.replace(codes)
    cates['score'] = score

    logger.info(
      f"ATE for treatment: {treatment}. outcome: {score} is "
      f"ate={ate} att={att}")
    results.append({
      'score': score, 'dropped': n_dropped,
      'ate': ate, 'att': att,  # 'cate': cate,
    })
    conditional_results.append(cates)
  return pd.DataFrame(results), pd.concat(conditional_results)


def main(_):
  exp_start = datetime.datetime.now()
  crime_bins = tuple(int(n) for n in FLAGS.crime_bins)
  ates, cates = average_treatment_effect(
    start_year=FLAGS.start_year,
    end_year=FLAGS.end_year,
    treatment='calc.race',
    binary_treatment_set={FLAGS.baseline: 0, FLAGS.treatment: 1},
    use_synth=FLAGS.synth,
    matching_alg=FLAGS.matching,
    repeat_match=FLAGS.repeat_match,
    n_subsample=FLAGS.n_subsample,
    crime_bins=crime_bins,
    seed=FLAGS.seed,
    # parameters to the synth generator
    window=FLAGS.window, lam=FLAGS.lam, omega=FLAGS.omega,
  )

  repo = git.Repo(search_parent_directories=True)
  custom_flags = FLAGS.get_key_flags_for_module(__file__)
  custom_flags = {f.name: f.value for f in custom_flags}
  custom_flags['commit'] = repo.head.commit.hexsha

  print(ates)
  print(custom_flags)

  data_path = BASE_DIR / 'data' / 'counterfact'
  data_path /= f'{FLAGS.start_year}-{FLAGS.end_year}_{FLAGS.window}'
  data_path /= f'{exp_start.month:02d}-{exp_start.day:02d}-{exp_start.hour:02d}'
  data_path.mkdir(parents=True, exist_ok=True)

  file_name = [
    'subsampled' if FLAGS.n_subsample is not None else '',
    f'{FLAGS.baseline}-{FLAGS.treatment}',
    ('synth' if FLAGS.synth else 'observed'),
    f'{FLAGS.matching}',
    'mrep' if FLAGS.repeat_match else 'nomrep',
  ]
  file_name += [] if not FLAGS.synth else [
    'nolam' if FLAGS.lam is None else f'lam3e{int(FLAGS.lam * 1000)}'
    f'om3e{int(FLAGS.omega * 1000)}',
    f'{FLAGS.seed}',
  ]
  file_name += ['b'.join(FLAGS.crime_bins[1:-1])]
  file_name = '_'.join(file_name)

  ates.to_csv(data_path / f'{file_name}-ate.csv', index=False)
  cates.to_csv(data_path / f'{file_name}-cate.csv', index=False)
  with open(data_path / f'{file_name}.json', 'w') as file:
    json.dump(custom_flags, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
  app.run(main)
