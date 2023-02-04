import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# pd.options.plotting.backend = 'matplotlib'


from cj_pipeline.config import BASE_DIR
from cj_pipeline.utils import subset_pd


def main():
  observed  = BASE_DIR / 'data/counterfact/1992-2012_2/02-04-10/subsampled_White-Black_observed_flame_nomrep_0b1b2b4b6b9b19b49-cate.csv'
  synth = BASE_DIR / 'data/counterfact/1992-2012_9/02-04-11/subsampled_White-Black_synth_flame_nomrep_lam3e1000_om3e1000_0_0b1b2b4b6b9b19b49-cate.csv'
  # synth = base_path / '02-01-18/subsampled_White-Black_synth_flame_nomrep_lam3e1000om3e1000_1_0b1b2b4b6b9b19b49-cate.csv'
  # observed = base_path / '1992-2012_2/02-04-10/subsampled_White-Black_observed_flame_nomrep_0b1b2b4b6b9b19b49-cate.csv'

  synth = pd.read_csv(synth)
  observed = pd.read_csv(observed)
  conditions = {
     'score': 'nca',  # 'nca', 'nvca', 'ogrs3', 'vprai', 'fta'
     'age_cat': '> 29',  # '> 29', '18-29'
     'def.gender': 'Female',  # 'Male', 'Female', '*'
  }

  sub_synth = subset_pd(synth, **conditions)
  sub_observed = subset_pd(observed, **conditions)

  weighted_avg_synth = np.average(sub_synth['cate'], weights=sub_synth['group_size'])
  weighted_avg_obs = np.average(sub_observed['cate'], weights=sub_observed['group_size'])

  print(weighted_avg_obs)
  print(weighted_avg_synth)


if __name__ == '__main__':
  main()
