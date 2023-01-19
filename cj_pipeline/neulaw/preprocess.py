import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm
import numpy as np
import uuid

from cj_pipeline.calculate_rais import calculate_rais
from cj_pipeline.neulaw.load import load
from cj_pipeline.config import logger

base_path = Path(__file__).parents[2] / 'data'


def init_rai_year_range(start_year: int, end_year: int):
    logger.info("Preparing Offence Counting...")
    df = load(base_path / 'neulaw')
    df = df[df["calc.year"] >= start_year]
    max_year = df["calc.year"].max()
    if end_year > max_year:
      logger.warning(f"Year {end_year} is greater than max year {max_year}")

    def get_risk_scores(year: int):
        logger.info(f"Counting Offences from {year} to {end_year}")
        if year > max_year:
            raise ValueError(f"Year {year} is greater than max year {max_year}")

        file_path = base_path / 'scratch' / f'rais_{year}-{end_year}.csv'
        if file_path.is_file():
          rais = pd.read_csv(file_path)
        else:
          count_df = preprocess(df, year, end_year)
          rais = calculate_rais(count_df)
          rais = rais[["def.uid", "nca", "nvca", "ogrs3", "vprai", "fta"]]
          rais.to_csv(file_path, index=False)

        return rais

    return get_risk_scores


def preprocess(
    df: pd.DataFrame, year_start:int=-1, year_end: int=np.inf
) -> pd.DataFrame:
    tqdm.pandas()

    logger.info("Starting Preprocessing..")
    logger.info("Converting generic operations")
    df = _generic_preprocessing(df, year_start=year_start, year_end=year_end)

    logger.info("Assign unique id")
    df = gen_unique_id(df)

    logger.info("Converting date columns")
    df = _conv_dates(df)

    logger.info("Processing age")
    df = _get_age(df)

    logger.info("Processing last case")
    df = _get_last_case(df)

    logger.info("Processing failed to appear")
    df = _get_fta(df)
    
    logger.info("Processing prior convictions")
    df = _get_priors(df)

    logger.info("Processing pending charges")
    df = _get_pending(df)

    logger.info("Get most serious offense")
    df = _get_most_serious_offense(df)

    logger.info("Processing current charge")
    df = _current_charge(df)
    logger.info("Processing criminal history")
    history = _get_criminal_history(df)
    logger.info("Done!")
    return history

# =============== Convert Dates ================= 

def _conv_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if 'date' in col:
            df[col] = pd.to_datetime(df[col])
    return df


def _generic_preprocessing(df: pd.DataFrame, year_start: int, year_end:int) -> pd.DataFrame:
    df["off.code"] = df["off.code"].fillna(0).astype(int).astype(str)
    df["disp.literal"] = df["disp.literal"].fillna("UNKNOWN")
    df = df[(df['calc.year'] >= year_start) & (df['calc.year'] <= year_end)]
    return df


def gen_unique_id(df) -> pd.DataFrame:
    df['uuid'] = df.apply(lambda _: uuid.uuid4(), axis=1)
    return df


# ==================== Calculate Age ====================

def _get_age(df: pd.DataFrame) -> pd.DataFrame:
    df['case.dt'] = pd.to_datetime(df['case.date'])
    df['def.dob.dt'] = pd.to_datetime(df['def.dob'])
    df['age'] = df['case.dt'] - df['def.dob.dt']
    df['age'] = df['age'].dt.days / 365.25
    df['age_cat'] = pd.cut(df['age'], bins=[0, 18, 31, 500], labels=['<18', '18-30', '31+'])
    return df

def _age_max(age_col: pd.Series) -> pd.Series:
    age_vals = {'<18': 0, '18-30': 1, '31+': 2}
    age_vals_list = ["<18", "18-30", "31+"]
    num_col = age_col.map(age_vals.get).max()
    if isinstance(num_col, float):
        return np.nan
    return age_vals_list[num_col]
    
# ==================== Calculate Failed To Appear ====================

def _count_offense(df: pd.DataFrame) -> pd.DataFrame:
    offense_types = ["assault", "drugs", "dui", "property", "sex offense"]
    def _to_colname(offense: str) -> str:
        return f"n_{offense.replace(' ', '_')}"
    def _n_offense(offense_type: str) -> int:
        return df[df['offense_category'] == offense_type].shape[0]
    offense_dict = {_to_colname(offense_type): _n_offense(offense_type) for offense_type in offense_types}
    return pd.DataFrame(offense_dict)

def _get_last_case(df: pd.DataFrame):
    """
    Get the last case date for each defendant
    """

    def _last(group):
        return group.idxmax()

    tqdm.pandas(desc='Getting last cases')
    # with logging_redirect_tqdm():#
    # only take with latest 'case.dt'
    # last_rows = df.groupby("def.uid")["case.dt"].idxmax()
    last_rows = df.groupby("def.uid")["case.dt"].progress_apply(_last)
    last_df = df.loc[last_rows]
    last_df = last_df.rename(
        {
            "calc.casenr": "last.case",
            "case.dt": "last.case.dt", 
            "off.date": "last.off.date"
        }, axis=1)
    last_df = last_df[["def.uid", "last.case", "last.case.dt", "last.off.date"]]
    df = df.merge(last_df, on="def.uid", how="left")
    df["diff"] = (df["last.off.date"] - df["off.date"]).dt.days
    return df


def _get_fta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get whether the defendant failed to appear within the last 2 years
    FROM THEIR LAST CASE (??) - Check this
    """
    offense_codes = ["501201", "501202", "820772", "820156", "500201", "500202"]

    prior_df = df[df["diff"] > 0][["uuid", "diff", "off.code"]]

    offense_cond = prior_df["off.code"].isin(offense_codes)

    prior_df["fta_lt_2yr"] = (offense_cond) & (prior_df["diff"] < 365.25 * 2)
    prior_df["fta_gt_2yr"] = (offense_cond) & (prior_df["diff"] >= 365.25 * 2)

    prior_df = prior_df.drop(["diff", "off.code"], axis=1)
    df = df.merge(prior_df, on=["uuid"], how="left")
    df = df.fillna({"fta_lt_2yr": False, "fta_gt_2yr": False})
    return df


# =================== Calculate Priors ===================


def _drug_conviction(row):
    if "Guilty" in row["calc.disp"] and row["offense_category"] == "drugs":
        return True
    return False

def _match_charge(tuple, charge, degree):
    return tuple[0] == charge and re.search(tuple[1], degree) is not None


def _violent_conviction(row, pending: bool = False, age_condition: bool = False):
    charge_degrees = [
        ("Arson", "F1"),
        ("Assault - Nonsexual", "F2|F3|FS|MA"),
        ("Burglary", "F1|F2"),
        ("Child Sex Crime", "F"),
        ("Crime Against Children", "F"),
        ("Evading/Resisting/Escaping", "F1|F2|F3"),
        ("Harassment/Stalking", "F2|F3"),
        ("Homicide|Kidnapping|Prostitution", "F1|F2|F3"),
        ("Sexual Assault")
    ]
    other_charges = "360112|110551|110549|110550|360115|360116|360111|110934|110533|110425|110625|110825|111025|111225|111425|110426|369998|369995|369996"
    if pending or "Guilty" in row["calc.disp"]:
        if not age_condition or row["age_cat"] != '<18':
            return re.search(other_charges, row["off.code"]) is not None or any([_match_charge(c, row["calc.broad"], row["case.degree"]) for c in charge_degrees])
    return False

def _incarceration(row):
    sentences = "COMMITTED TO LOCAL JAIL|COMMITTED TO TDC|STATE JAIL|LIFE SENTENCE|SHOCK PROBATION"
    return re.search(sentences, row["disp.literal"]) is not None

def _get_priors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the number of priors for each defendant
    """
    prior_df = df[(df["diff"] > 0) | (df["diff"].isna())].copy()

    # Priors
    tqdm.pandas(desc='Counting drug convictions')
    prior_df["drug.conviction"] = prior_df.progress_apply(_drug_conviction, axis=1)

    tqdm.pandas(desc='Counting violent convictions')
    prior_df["violent.conviction"] = prior_df.progress_apply(_violent_conviction, axis=1)

    tqdm.pandas(desc='Counting violent convictions (adult)')
    prior_df["violent.conviction.adult"] = prior_df.progress_apply(_violent_conviction, age_condition=True, axis=1)
    
    tqdm.pandas(desc='Counting incarcerations')
    prior_df["incarceration"] = prior_df.progress_apply(_incarceration, axis=1)

    prior_df["conviction"] = prior_df["calc.disp"].str.contains("Guilty")

    prior_df["not.dismissed"] = prior_df["calc.disp"] != "Dismissal"

    prior_df["misdemeanor"] = prior_df["case.degree"].str.contains("M")

    prior_df["felony"] = prior_df["case.degree"].str.contains("F")

    prior_df = prior_df[[
        'uuid',
        'drug.conviction', 
        'violent.conviction', 
        'violent.conviction.adult', 
        'incarceration', 
        'conviction', 
        'not.dismissed', 
        'misdemeanor', 
        'felony']]

    df = df.merge(prior_df, on="uuid", how="left")

    df = df.fillna({
        "drug.conviction": False,
        "violent.conviction": False,
        "violent.conviction.adult": False,
        "incarceration": False,
        "conviction": False,
        "not.dismissed": False,
        "misdemeanor": False,
        "felony": False
    })

    return df

#  =================== Calculate Pending Charges ===================

def _get_pending(df: pd.DataFrame) -> pd.DataFrame:

    # Only prior cases
    prior_df = df[(df["diff"] > 0) | (df["diff"].isna())].copy()
    # Only pending cases
    pending_df = prior_df[(prior_df["last.off.date"] - prior_df["disp.date"]).dt.days < 0]


    pending_df["violent.pending"] = pending_df.progress_apply(_violent_conviction, pending=True, axis=1)
    pending_df["pending.charge"] = True

    pending_df = pending_df[["uuid", "violent.pending", "pending.charge"]]

    df = df.merge(pending_df, on="uuid", how="left")
    df["pending.charge"] = df["pending.charge"].fillna(False)
    df["violent.pending"] = df["violent.pending"].fillna(False)

    return df

def _current_charge(df: pd.DataFrame) -> pd.DataFrame:

    current_df = df[df["diff"] == 0].copy()

    current_df["current.felony"] = current_df["case.degree"].str.contains("F")
    current_df["current.violent"] = current_df.apply(_violent_conviction, pending=True, axis=1)
    current_df["current.conviction"] = current_df["calc.disp"].str.contains("Guilty")
    current_df["current.age.numeric"] = current_df["age"]

    current_df = current_df[[
        "uuid", 
        "current.felony", 
        "current.violent", 
        "current.conviction",
        "current.age.numeric",
    ]]

    df = df.merge(current_df, on=["uuid"], how="left")

    df["current.felony"] = df["current.felony"].fillna(False)
    df["current.violent"] = df["current.violent"].fillna(False)
    df["current.conviction"] = df["current.conviction"].fillna(False)

    return df


#  =================== Calculate Criminal History ===================

def _get_most_serious_offense_degree(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the most serious offense for each defendant
    """
    def _degree(degree):
        match degree:
            case _ if degree == "M":
                return 2
            case _ if degree == "MA":
                return 3
            case _ if degree == "MB":
                return 2
            case _ if degree == "MC":
                return 1
            case _ if degree == "FS":
                return 4
            case _ if degree == "F3":
                return 5
            case _ if degree == "F2":
                return 6
            case _ if degree == "F1":
                return 7
            case _ if degree == "FC":
                return 8
            case _ if degree == "F":
                return 6
            case _:
                return 0
    tqdm.pandas(desc='Calculating degree of offense')
    df["degree_num"] = df['case.degree'].progress_apply(_degree)
    return df

def _get_most_serious_offense(df: pd.DataFrame):
    groups = ['def.gender', 'def.race', 'def.uid']
    df = _get_most_serious_offense_degree(df)
    def _most_serious_offense(group):
            return group["calc.detailed"][group["degree_num"] == group["degree_num"].max()].iloc[0]
    srs_off = df.groupby(groups).progress_apply(_most_serious_offense)
    srs_off = srs_off.to_frame("most_serious_offense").reset_index()
    df = df.merge(srs_off, on=groups, how="left")
    return df


def _get_criminal_history(df: pd.DataFrame) -> pd.DataFrame:
    groups = ['def.gender', 'def.race', 'def.uid']
    agg = df.groupby(groups).agg(
        {
            "age_cat": _age_max,
            "case.dt": "max",
            "fta_lt_2yr": "sum",
            "fta_gt_2yr": "sum",
            "drug.conviction": "sum",
            "violent.conviction": "sum",
            "violent.conviction.adult": "sum",
            "incarceration": "sum",
            "conviction": "sum",
            "not.dismissed": "sum",
            "misdemeanor": "sum",
            "felony": "sum",
            "age": "min",
            "violent.pending": "sum",
            "pending.charge": "sum",
            "current.felony": "first",
            "current.violent": "first",
            "current.conviction": "first",
            "current.age.numeric": "first",
            "most_serious_offense": "first"
        }
    ).reset_index()
    # offenses = df.groupby(groups).apply(_count_offense)
    # df = agg.merge(offenses, on='def.uid', how='left')

    agg = agg.rename({
        "age": "age_first_arrest",
        "case.dt": "last_arrest_date",
        "age_cat": "current_age",
        "fta_lt_2yr": "fta_lt_2yr_count",
        "fta_gt_2yr": "fta_gt_2yr_count",
        "drug.conviction": "drug_conviction_count",
        "violent.conviction": "violent_conviction_count",
        "violent.conviction.adult": "violent_conviction_adult_count",
        "incarceration": "incarceration_count",
        "conviction": "conviction_count",
        "not.dismissed": "not_dismissed_count",
        "misdemeanor": "misdemeanor_count",
        "felony": "felony_count",
        "violent.pending": "violent_pending_count",
        "pending.charge": "pending_charge_count",
        "calc.detailed": "most_serious_offense"
        }, axis=1)
    # df["age.first.arrest"] = df["age.first.arrest"].astype(int) / 365.25

    agg = agg[agg["def.gender"] != "Missing"]
    agg = agg[agg["def.race"] != "Missing"]
    agg = agg.dropna(subset=["age_first_arrest"], axis=0)

    return agg

