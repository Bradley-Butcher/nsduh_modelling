from cj_pipeline.ncvs.load import load
import pandas as pd
from tqdm import tqdm
from cj_pipeline.config import logger
import numpy as np

def preprocess() -> pd.DataFrame:
    df = load()
    logger.info(f"Preprocessing data")
    logger.info(f"Processing crime type")
    df = _process_crime_type(df)
    logger.info(f"Processing offender race")
    df = _process_offender_race(df)
    logger.info("Processing offender age")
    df = _process_offender_age(df)
    logger.info("Processing offender sex")
    df = _process_offender_sex(df)
    logger.info("Processing reported to police")
    df = _process_reported_to_police(df)
    logger.info("Processing arrest or charges made")
    df = _process_arrests_or_charges_made(df)
    logger.info("Summarizing")
    df = _summarize(df)
    return df

def _process_crime_type(df: pd.DataFrame) -> pd.DataFrame:
    def _crime_type(crime_type):
        if crime_type in [
            "(01) Completed rape",
            "(02) Attempted rape",
            "(03) Sex aslt w s aslt",
            "(04) Sex aslt w m aslt",
            "(15) Sex aslt wo inj",
            "(16) Unw sex wo force"
            ]:
            return "sex offense"
        elif crime_type in [
            "(05) Rob w inj s aslt",
            "(06) Rob w inj m aslt",
            "(07) Rob wo injury",
            "(08) At rob inj s asl",
            "(09) At rob inj m asl",
            "(10) At rob w aslt"
            ]:
            return "robbery"
        elif crime_type in [
            "(11) Ag aslt w injury",
            "(12) At ag aslt w wea",
            "(13) Thr aslt w weap"
            ]:
            return "aggravated assault"
        elif crime_type in [
            "(14) Simp aslt w inj",
            "(17) Asl wo weap, wo inj",
            "(20) Verbal thr aslt"
            ]:
            return "simple assault"
        elif crime_type in [
            "(21) Purse snatching",
            "(22) At purse snatch",
            "(23) Pocket picking",
            "(40) Motor veh theft",
            "(59) Attempted theft",
            "(41) At mtr veh theft",
            "(23) Pocket picking",
            "(54) Theft < $10",
            "(55) Theft $10-$49",
            "(56) Theft $50-$249",
            "(57) Theft $250+",
            "(58) Theft value NA",
            "(59) Attempted theft"
            ]:
            return "property"
        elif crime_type in [
            "(31) Burg, force ent",
            "(32) Burg, ent wo for",
            "(33) Att force entry"
            ]:
            return "property"
        else:
            return "other"

    tqdm.pandas(desc='Processing Crime Type')
    df["crime_recode"] = df["crime_type"].progress_apply(_crime_type)
    df = df[df["crime_recode"] != "other"]
    return df

def _process_offender_race(df: pd.DataFrame) -> pd.DataFrame:
    def _offender_race(row):
        if row["c_mult_off_race_black"] == "(1) Yes":
            return "Black"
        elif row["c_mult_off_race_white"] == "(1) Yes":
            return "White"
        elif row["single_offender_race_end_2011_q4"] == "(1) White":
            return "White"
        elif row["single_offender_race_end_2011_q4"] == "(2) Black":
            return "Black"
        elif row["multiple_offender_race_of_most_end_2011_q4"] == "(1) Mostly White":
            return "White"
        elif row["multiple_offender_race_of_most_end_2011_q4"] == "(2) Mostly Black":
            return "Black"
        elif row["multiple_offender_race_of_most_start_2012_q1"] == "(1) Mostly White":
            return "White"
        elif row["multiple_offender_race_of_most_start_2012_q1"] == "(2) Mostly Black":
            return "Black"
        elif row["c_single_offender_race_white_start_2012_q1"] == "(1) Yes":
            return "White"
        elif row["c_single_offender_race_black_or_african_american_start_2012_q1"] == "(1) Yes":
            return "Black"
        else:
            return "Other"
    tqdm.pandas(desc='Processing Offender Race')
    df["offender_race"] = df.progress_apply(_offender_race, axis=1)
    return df


def _process_offender_age(df: pd.DataFrame) -> pd.DataFrame:
    def _offender_age(row):
        if row["single_offender_age"] == "(1) Under 12":
            return "< 18"
        elif row["single_offender_age"] == "(2) 12-14":
            return "< 18"
        elif row["single_offender_age"] == "(3) 15-17":
            return "< 18"
        elif row["single_offender_age"] == "(4) 18-20":
            return "18-30"
        elif row["single_offender_age"] == "(5) 21-29":
            return "18-30"
        elif row["single_offender_age"] == "(6) 30+":
            return "> 30"
        elif (row["multiple_offenders_age_of_oldest"] == row["multiple_offenders_age_of_youngest"]) and row["multiple_offenders_age_of_oldest"] == "(1) Under 12":
            return "< 18"
        elif (row["multiple_offenders_age_of_oldest"] == row["multiple_offenders_age_of_youngest"]) and row["multiple_offenders_age_of_oldest"] == "(2) 12-14":
            return "< 18"
        elif (row["multiple_offenders_age_of_oldest"] == row["multiple_offenders_age_of_youngest"]) and row["multiple_offenders_age_of_oldest"] == "(3) 15-17":
            return "< 18"
        elif (row["multiple_offenders_age_of_oldest"] == row["multiple_offenders_age_of_youngest"]) and row["multiple_offenders_age_of_oldest"] == "(4) 18-20":
            return "18-30"
        elif (row["multiple_offenders_age_of_oldest"] == row["multiple_offenders_age_of_youngest"]) and row["multiple_offenders_age_of_oldest"] == "(5) 21-29":
            return "18-30"
        elif (row["multiple_offenders_age_of_oldest"] == row["multiple_offenders_age_of_youngest"]) and row["multiple_offenders_age_of_oldest"] == "(6) 30+":
            return "> 30"
        else:
            return "other"
    tqdm.pandas(desc='Processing Offender Age')
    df["offender_age"] = df.progress_apply(_offender_age, axis=1)
    return df


def _process_offender_sex(df: pd.DataFrame) -> pd.DataFrame:
    def _offender_sex(row):
        if row["single_offender_sex"] == "(1) Male":
            return "Male"
        elif row["single_offender_sex"] == "(2) Female":
            return "Female"
        elif row["multiple_offenders_sex"] == "(1) All male":
            return "Male"
        elif row["multiple_offenders_sex"] == "(2) All female":
            return "Female"
        elif row["multiple_offenders_mostly_male_or_female"] == "(1) Mostly male":
            return "Male"
        elif row["multiple_offenders_mostly_male_or_female"] == "(2) Mostly female":
            return "Female"
        else:
            return "Other"
    tqdm.pandas(desc='Processing Offender Sex')
    df["offender_sex"] = df.progress_apply(_offender_sex, axis=1)
    return df


def _process_reported_to_police(df: pd.DataFrame) -> pd.DataFrame:
    def _reported_to_police(reported_to_police):
        if reported_to_police == "(1) Yes":
            return 1
        elif reported_to_police == "(2) No":
            return 0
        else:
            return np.nan
    tqdm.pandas(desc='Processing Reported to the Police')
    df["reported_to_police"] = df["reported_to_police"].progress_apply(_reported_to_police)
    return df

def _process_arrests_or_charges_made(df: pd.DataFrame) -> pd.DataFrame:
    def _arrests_or_charges_made(row):
        if row["arrests_or_charges_made"] == "(1) Yes":
            return 1
        elif row["arrests_or_charges_made"] == "(2) No":
            return 0
        elif row["arrests_or_charges_made"] == "(9) Out of universe":
            return 0
        elif row["reported_to_police"] == 0:
            return 0
        else:
            return np.nan
    tqdm.pandas(desc='Processing arrests or charges made')
    df["arrests_or_charges_made"] = df.progress_apply(_arrests_or_charges_made, axis=1)
    df = df[df["arrests_or_charges_made"].notnull()]
    return df

def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(
        [
            "crime_recode",
            "offender_age",
            "offender_sex",
            "offender_race"
        ]).agg({
        "arrests_or_charges_made": "mean",
        "reported_to_police": "mean",
        }).reset_index()
    count = df.groupby(
        [
            "crime_recode",
            "offender_age",
            "offender_sex",
            "offender_race"
        ]).size().to_frame("count").reset_index()
    agg.rename(columns={"arrests_or_charges_made": "arrest_rate", "reported_to_police": "reporting_rate"}, inplace=True)
    df = pd.merge(agg, count, on=["crime_recode", "offender_age", "offender_sex", "offender_race"])
    return df
