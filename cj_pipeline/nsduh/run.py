from cj_pipeline.nsduh.load import load_nsduh

from pathlib import Path

from cj_pipeline.nsduh.preprocess import preprocess

if __name__ == "__main__":
    base_path = Path(__file__).parents[2] / 'data' / 'nsduh'
    nsduh = load_nsduh()
    nsduh = preprocess(nsduh)
    nsduh.to_csv(base_path.parent / "processed" / 'nsduh.csv', index=False)
