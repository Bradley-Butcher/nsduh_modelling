import pandas as pd
from absl import app, flags

from cj_pipeline.neulaw.assignment_preprocessing import init_neulaw
from cj_pipeline.neulaw.preprocess import init_rai_year_range
from cj_pipeline.config import logger, BASE_DIR, CRIMES

import dame_flame

SCORES = [
  'nca',
  'nvca',
  'ogrs3',
  'vprai',
  'fta'
]
DEMOGRAPHICS = [
  "def.race",
  "def.gender",
  "age_cat",
]
MATCHING_ALGS = [
  'dame',
  'flame',
  'hybrid',
]


FLAGS = flags.FLAGS
flags.DEFINE_bool('synth', False, 'Use synthetic data.')
flags.DEFINE_integer('start_year', 1992, 'Initial year of records')
flags.DEFINE_integer('window', 20, 'No. of years to include after `start_year`.')
flags.DEFINE_enum('matching', 'flame', MATCHING_ALGS, help=f'One of {MATCHING_ALGS}.')


def _min_max_scale(df: pd.DataFrame) -> pd.DataFrame:
  return (df - df.min()) / (df.max() - df.min())


def _binarize_treatment(
    df: pd.DataFrame,
    treatment: str,
    binary_treatment_set: dict[str, int]
) -> pd.DataFrame:
  assert len(binary_treatment_set) == 2, "binary_treatment_set must have 2 keys"

  df = df.copy()
  if "Rest" in binary_treatment_set.keys():
    values = df[treatment].unique()
    other = set(values) - set(binary_treatment_set.keys())
    for o_i in other:
      binary_treatment_set[o_i] = binary_treatment_set["Rest"]
  df = df[df[treatment].isin(list(binary_treatment_set.keys()))]
  df[treatment] = df[treatment].map(binary_treatment_set)
  return df


def _binarize_crimes(df: pd.DataFrame):
  for crime in CRIMES:
    df[crime] = pd.Categorical(
      pd.cut(df[crime], right=True, bins=[-1, 1, 2, 9, 100_000])).codes
  return df


# TODO: sensitivity analysis wrt params of the matching algorithms?
def _matching_model(score_df, matching_alg):
  matching_alg = matching_alg.lower()
  if matching_alg not in MATCHING_ALGS:
    raise ValueError(f'Unknown matching algorithm "{matching_alg}"')

  if matching_alg == 'dame':
    model = dame_flame.matching.DAME(repeats=True)
  else:
    model = dame_flame.matching.FLAME(repeats=True)
  kwargs = {'pre_dame': 1} if matching_alg == 'hybrid' else {}

  model.fit(score_df)
  _ = model.predict(score_df, **kwargs)

  return model


def average_treatment_effect(
    start_year: int,
    window: int,
    treatment: str,
    binary_treatment_set: dict[str, int],
    use_synth: bool = False,
    matching_alg: str = 'flame',
) -> pd.DataFrame:
  logger.info('Loading RAI scores')
  rai_func = init_rai_year_range(start_year=start_year, window=window)
  rai_dfs = rai_func(start_year)

  logger.info('Loading offense counts')
  if use_synth:
    logger.info('Estimating based on synthethic crime record')
    synth_data_path = BASE_DIR / 'data' / 'processed' / 'synth_crimes.csv'
    offense_dfs = pd.read_csv(synth_data_path)
  else:
    logger.info('Estimating based on recorded crime only')
    offense_count_func, _ = init_neulaw(
      start_year=start_year, window=window, melt=False)
    offense_dfs = offense_count_func(start_year)

  df = pd.merge(offense_dfs, rai_dfs, on=['def.uid'])
  df = df[SCORES + CRIMES + DEMOGRAPHICS]
  df = _binarize_crimes(df)
  df = _binarize_treatment(df, treatment, binary_treatment_set)
  for dem in set(DEMOGRAPHICS) - set([treatment]):
    df[dem] = pd.Categorical(df[dem]).codes

  all_score_df = _min_max_scale(df[SCORES])
  df = df.drop(columns=SCORES)
  df = df.rename(columns={treatment: 'treated'})

  results = []
  for score in SCORES:
    logger.info(f"Calculating ATE for treatment: {treatment}. outcome: {score}")
    score_df = df.copy()

    score_df["outcome"] = all_score_df[score]
    before_len = len(score_df)
    score_df = score_df.dropna(subset=["outcome"])
    after_len = len(score_df)
    logger.info(
      f"Dropped {before_len - after_len} rows with missing values for {score}")

    model = _matching_model(score_df, matching_alg=matching_alg)
    ate = dame_flame.utils.post_processing.ATE(matching_object=model)
    logger.info(f"ATE for treatment: {treatment}. outcome: {score} is {ate}")

    results.append({'score': score, 'ate': ate})
  return pd.DataFrame(results)


def main(_):
  df = average_treatment_effect(
    start_year=FLAGS.start_year,
    window=FLAGS.window,
    treatment='def.race',
    binary_treatment_set={"Black": 1, "White": 0},
    use_synth=FLAGS.synth,
    matching_alg=FLAGS.matching
  )

  year_range = f'{FLAGS.start_year}-{FLAGS.start_year + FLAGS.window}'
  file_name = f'{FLAGS.matching}_{year_range}'
  file_name += '_synth' if FLAGS.synth else ''
  df.to_csv(BASE_DIR / 'data' / 'counterfact' / file_name, index=False)

  print(df)


if __name__ == "__main__":
  app.run(main)
