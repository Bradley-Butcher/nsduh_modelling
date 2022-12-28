import pandas as pd
from cj_pipeline.ogrs3_coefs import get_ogrs3_conviction_coef, ogrs3_gender_coef
from config import logger
from pathlib import Path
from tqdm import tqdm
import numpy as np

data_path = Path(__file__).parents[1] / 'data'

def load_criminal_history():
    tqdm.pandas()
    logger.info('Loading criminal history')
    return pd.read_csv(data_path / "processed" / "criminal_history.csv")

def calculate_rais(df: pd.DataFrame):
    logger.info('Calculating Risk Assesment Instruments')
    logger.info('Calculating NCA')
    df = get_nca(df)
    logger.info('Calculating NVCA')
    df = get_nvca(df)
    logger.info('Calculating OGRS3')
    df = get_ogrs3(df)
    return df

def get_nca(df: pd.DataFrame):
    def _calc_nca(row):
        score = 0
        score += 0 if row['current.age.numeric'] >= 23 else 2
        score += 3 if row['pending_charge_count'] > 0 else 0
        score += 1 if row['misdemeanor_count'] > 0 else 0
        score += 1 if row['felony_count'] > 0 else 0
        score += 2 if row['violent_conviction_count'] >= 3 else 1 if row['violent_conviction_count'] > 0 else 0
        score += 2 if row['fta_lt_2yr_count'] >= 2 else 1 if row['fta_lt_2yr_count'] > 0 else 0
        score += 2 if row['incarceration_count'] > 0 else 0
        match score:
            case _ if score == 0: return 1
            case _ if score <= 2: return 2
            case _ if score <= 4: return 3
            case _ if score <= 6: return 4
            case _ if score <= 8: return 5
            case _ if score <= 13: return 6
            case _: return np.nan
    tqdm.pandas(desc='Calculating NCA')
    df['nca'] = df.progress_apply(_calc_nca, axis=1)
    return df

def get_nvca(df: pd.DataFrame):
    def _calc_nvca(row):
        score = 0
        score += 2 if row['violent_pending_count'] > 0 else 0
        score += 1 if row['violent_pending_count'] == 1 and row['current.age.numeric'] < 21 else 0
        score += 1 if row['pending_charge_count'] > 0 else 0
        score += 1 if row['conviction_count'] > 0 else 0
        score += 1 if row['felony_count'] > 0 else 0
        score += 2 if row['violent_conviction_count'] >= 3 else 1 if row['violent_conviction_count'] > 0 else 0
        match score:
            case _ if score <= 1: return 1
            case _ if score <= 2: return 2
            case _ if score <= 3: return 3
            case _ if score <= 4: return 4
            case _ if score <= 5: return 5
            case _ if score <= 7: return 6
            case _: return np.nan
    tqdm.pandas(desc='Calculating NVCA')
    df['nvca'] = df.progress_apply(_calc_nvca, axis=1)
    return df

def get_fta(df: pd.DataFrame):
    def _calc_fta(row):
        score = 0
        score += 1 if row['pending_charge_count'] > 0 else 0
        score += 1 if row['conviction_count'] > 0 else 0
        score += 4 if row['fta_lt_2yr_count'] >= 2 else 2 if row['fta_lt_2yr_count'] > 0 else 0
        score += 1 if row['fta_gt_2yr_count'] > 0 else 0
        match score:
            case _ if score == 0: return 1
            case _ if score == 1: return 2
            case _ if score == 2: return 3
            case _ if score <= 4: return 4
            case _ if score <= 6: return 5
            case _ if score <= 7: return 6
    tqdm.pandas(desc='Calculating FTA')
    df['fta'] = df.progress_apply(_calc_fta, axis=1)
    return df

def get_vprai(df: pd.DataFrame):
    def _calc_vprai(row):
        score = 0
        score += 1 if row['felony_count'] > 0 else 0
        score += 1 if row['pending_charge_count'] > 0 else 0
        score += 1 if row['conviction_count'] > 0 else 0
        score += 2 if row['fta_lt_2yr_count'] + row['fta_gt_2yr_count'] >= 2 else 0
        score += 1 if row['violent_conviction_adult_count'] >= 2 else 0
        score += 1 if row['drug_conviction_count'] > 0 else 0
        return score
    tqdm.pandas(desc='Calculating VPR-AI')
    df['vprai'] = df.progress_apply(_calc_vprai, axis=1)
    return df
    
def get_ogrs3(df: pd.DataFrame):
    coefs = pd.read_csv(data_path / "rais" / "coef_ogrs3.csv")
    df = df.merge(coefs, left_on='most_serious_offense', right_on='calc.detailed', how='left')
    def _calc_ogrs3(row):
        score = 2.121705678
        score += row['coef']
        score += get_ogrs3_conviction_coef(row)
        denom = 10 + (0 if row['age_first_arrest'] == 0 else row['current.age.numeric'] - np.floor(row['age_first_arrest']))
        score += np.log((row['not_dismissed_count'] + 1) / denom) * 1.251124464
        score += ogrs3_gender_coef(row)
        return 1/(1 + np.exp(-score))
    tqdm.pandas(desc='Calculating OGRS3')
    df['ogrs3'] = df.progress_apply(_calc_ogrs3, axis=1)
    df.drop(columns=['coef'], inplace=True)
    return df


if __name__ == "__main__":
    history = load_criminal_history()
    rais = calculate_rais(history)
    rais.to_csv(data_path / "rais" / "calculated_rais.csv", index=False)
