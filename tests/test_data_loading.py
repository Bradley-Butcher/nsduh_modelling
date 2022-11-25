from cj_pipeline.load import load_nsduh

def test_load():
    df = load_nsduh(max_rows=1000)
    assert len(df) <= 1000
    assert len(df.columns) == 12

    
    
