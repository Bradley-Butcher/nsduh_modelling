from pathlib import Path
from cj_pipeline.ncvs.load import load
from cj_pipeline.ncvs.preprocess import preprocess

if __name__ == "__main__":
    df = load()
    df = preprocess(df)
    df.to_csv(Path(__file__).parents[2] / 'data' / 'processed' / 'ncvs.csv', index=False)
