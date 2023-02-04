from pathlib import Path
from cj_pipeline.nsduh.load import load_nsduh
from cj_pipeline.nsduh.preprocess import preprocess


def main():
  save_path = Path(__file__).parents[2] / 'data' / 'processed'
  nsduh = load_nsduh()
  nsduh_dicts = preprocess(nsduh)
  for mode in nsduh_dicts:
    nsduh_dicts[mode].to_csv(save_path / f'nsduh_{mode}.csv', index=False)


if __name__ == "__main__":
  main()
