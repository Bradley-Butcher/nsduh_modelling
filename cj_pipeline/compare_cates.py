import pandas as pd
# import matplotlib.pyplot as plt
# pd.options.plotting.backend = 'matplotlib'

from cj_pipeline.config import BASE_DIR
from cj_pipeline.utils import subset_pd


def main():
  base_path = BASE_DIR / 'data' / 'counterfact'
  synth = base_path / '1992-2012_2/02-01-11/subsampled_White-Black_synth_flame_nomrep_nolam_0_0b1b2b4b9-cate.csv'
  observed = base_path / '1992-2012_2/02-01-10/subsampled_White-Black_observed_flame_nomrep_0b1b2b4b9-cate.csv'

  synth = pd.read_csv(synth)
  observed = pd.read_csv(observed)
  conditions = {
    'score': 'nca',  # 'nca', 'nvca', 'ogrs3', 'vprai', 'fta'
    'age_cat': '> 29',  # '> 29', '18-29'
    'def.gender': 'Male',  # 'Male', 'Female', '*'
  }

  sub_synth = subset_pd(synth, **conditions)
  sub_observed = subset_pd(observed, **conditions)

  print('done')


if __name__ == '__main__':
  main()
