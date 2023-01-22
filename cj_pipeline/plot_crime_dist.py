import pandas as pd
import matplotlib.pyplot as plt
pd.options.plotting.backend = 'matplotlib'

from cj_pipeline.neulaw.assignment_preprocessing import init_neulaw
from cj_pipeline.synthetic_assignment import get_synth
from cj_pipeline.config import CRIMES


def main():
  start_year, end_year = 1992, 2012

  # crime_bins = crime_bins,
  # parameters to the synth generator
  window = 2
  lam = 1.0
  omega = 1.0
  seed = 0

  # load non synth
  neulaw_gen, _ = init_neulaw(
    start_year=start_year, window=end_year - start_year, melt=False)
  non_synth = neulaw_gen(start_year)

  # load synth
  synth = get_synth(
    start_year=start_year, end_year=end_year, window=window,
    lam=lam, omega=omega, seed=seed)

  def _plot(df, name):
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for a, crime in zip(ax.flat, CRIMES):
      for race in df['def.race'].unique():
        a.hist(df[df['def.race'] == race][crime], 20, log=True, label=race, alpha=0.5)
      a.set_title(crime)

    ax[0, 0].legend()
    fig.tight_layout()
    fig.suptitle(name, y=0.15)

  _plot(synth, 'synth')
  _plot(non_synth, 'non synth')



if __name__ == '__main__':
  main()
