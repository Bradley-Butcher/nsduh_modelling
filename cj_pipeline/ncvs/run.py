from pathlib import Path
from cj_pipeline.ncvs.load import load
from cj_pipeline.ncvs.preprocess import preprocess


def main():
  save_path = Path(__file__).parents[2] / 'data' / 'processed'
  ncvs = load()
  ncvs_dicts = preprocess(ncvs)
  for mode in ncvs_dicts:
    ncvs_dicts[mode].to_csv(save_path / f'ncvs_{mode}.csv', index=False)


if __name__ == "__main__":
  main()
