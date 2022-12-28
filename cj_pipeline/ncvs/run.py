from pathlib import Path
from cj_pipeline.ncvs.preprocess import preprocess

if __name__ == "__main__":
    df = preprocess()
    df.to_csv(Path(__file__).parents[2] / 'data' / 'processed' / 'ncvs.csv', index=False)
