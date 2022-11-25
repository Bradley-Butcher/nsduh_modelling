from cj_pipeline.neulaw.load import load
from cj_pipeline.neulaw.preprocess import preprocess

from pathlib import Path

if __name__ == "__main__":
    base_path = Path(__file__).parents[2] / 'data' / 'neulaw'
    harrod_county = load(base_path)
    harrod_processed = preprocess(harrod_county)
    harrod_processed.to_csv(base_path.parent / "processed" / 'criminal_history.csv', index=False)
