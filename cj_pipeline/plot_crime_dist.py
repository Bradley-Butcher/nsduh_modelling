import pandas as pd
import matplotlib.pyplot as plt
pd.options.plotting.backend = 'matplotlib'

from cj_pipeline.neulaw.assignment_preprocessing import init_neulaw
from cj_pipeline.neulaw.preprocess import init_rai_year_range
from cj_pipeline.synthetic_assignment import get_synth
from cj_pipeline.config import CRIMES


def main():
  start_year, end_year = 1992, 2012
  race_col = 'calc.race'  # 'def.race'

  # parameters to the synth generator
  window = 2
  omega = 1.0
  seed = 0

  # load non synth
  neulaw_gen, _ = init_neulaw(
    start_year=start_year, window=end_year - start_year)
  non_synth = neulaw_gen(start_year)

  # load synth
  synth = get_synth(
    start_year=start_year, end_year=end_year, window=window,
    omega=omega, seed=seed)

  # load rais
  rai_func = init_rai_year_range(start_year=start_year, end_year=end_year)
  rai_dfs = rai_func(start_year)

  synth = pd.merge(synth, rai_dfs, on=['def.uid'])
  non_synth = pd.merge(non_synth, rai_dfs, on=['def.uid'])

  def _plot(df, name):
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for a, crime in zip(ax.flat, CRIMES):
      for race in df[race_col].unique():
        a.hist(df[df[race_col] == race][crime], 20, log=True, label=race, alpha=0.5)
      a.set_title(crime)

    ax[0, 0].legend()
    fig.tight_layout()
    fig.suptitle(name, x=0.85, y=0.15)

  idx_synth = synth['nca'] == 1
  idx_nonsynth = non_synth['nca'] == 1

  _plot(synth[idx_synth], 'synth')
  _plot(non_synth[idx_nonsynth], 'non synth')
  plt.show()



if __name__ == '__main__':
  main()
