import pandas as pd
from functools import partial


def process_bkdrug(df: pd.DataFrame):
    df["BKDRUG"] = df["BKDRUG"].astype("int")
    d = {
        1: "Yes",
        2: "No",
        3: "Yes",
        99: "No"
    }
    df['BKDRUG'] = df['BKDRUG'].map(d)
    df = df.dropna(subset=['BKDRUG'], axis=0)
    return df


def process_drug(df, drug_col):
    df[drug_col] = df[drug_col].astype("int")
    df.loc[df[drug_col] > 365, drug_col] = 0
    return df


def process_catag6(df):
    df['CATAG6'] = df['CATAG6'].astype("int")
    d = {
        1: "12-17",
        2: "18-25",
        3: "26-34",
        4: "35-49",
        5: "50-64",
        6: "65+"
    }
    df['CATAG6'] = df['CATAG6'].map(d)
    df = df.dropna(subset=['CATAG6'], axis=0)
    return df


def process_newrace2(df):
    df['NEWRACE2'] = df['NEWRACE2'].astype("int")
    d = {
        1: "White",
        2: "Black",
        7: "Hispanic",
    }
    df['NEWRACE2'] = df['NEWRACE2'].map(d)
    df = df.dropna(subset=['NEWRACE2'], axis=0)
    return df


def process_eduhighcat(df):
    df['EDUHIGHCAT'] = df['EDUHIGHCAT'].astype("int")
    d = {
        1: "<High School",
        2: "High School",
        3: "Some College",
        4: "College Graduate",
        5: "12-17 years old",
    }
    df['EDUHIGHCAT'] = df['EDUHIGHCAT'].map(d)
    df = df.dropna(subset=['EDUHIGHCAT'], axis=0)
    return df


def process_irsex(df):
    df['IRSEX'] = df['IRSEX'].astype("int")
    d = {
        1: "Male",
        2: "Female"
    }
    df['IRSEX'] = df['IRSEX'].map(d)
    df = df.dropna(subset=['IRSEX'], axis=0)
    return df

def process_snysell(df):
    df['SNYSELL'] = df['SNYSELL'].astype("int")
    d = {
        1: "0 times",
        2: "1-2 times",
        3: "3-5 times",
        4: "6-9 times",
        5: "10+ times",
        99: "0 times"
    }
    df['SNYSELL'] = df['SNYSELL'].map(d)
    df = df.dropna(subset=['SNYSELL'], axis=0)
    return df

variable_pp = {
    "BKDRUG": process_bkdrug,
    "IRMJFY": partial(process_drug, drug_col="IRMJFY"),
    "IRCOCFY": partial(process_drug, drug_col="IRCOCFY"),
    "IRHERFY": partial(process_drug, drug_col="IRHERFY"),
    "IRCRKFY": partial(process_drug, drug_col="IRCRKFY"),
    "CATAG6": process_catag6,
    "NEWRACE2": process_newrace2,
    "EDUHIGHCAT": process_eduhighcat,
    "IRSEX": process_irsex,
    "SNYSELL": process_snysell
}

drug_vars  = ["IRMJFY", "IRCOCFY", "IRHERFY", "IRCRKFY"]

def total_drug_use(df):
    df['Total Drug Use'] = df[drug_vars].sum(axis=1)
    return df

variable_names = {
    "BKDRUG": "Arrested",
    "IRMJFY": "Marijuana Past Year",
    "IRCOCFY": "Cocaine Past Year",
    "IRHERFY": "Heroin Past Year",
    "IRCRKFY": "Crack Past Year",
    "CATAG6": "Age",
    "NEWRACE2": "Race",
    "EDUHIGHCAT": "Education",
    "IRSEX": "Sex",
    "SNYSELL": "Sold Drugs Past Year"
}


def get_variables():
    return list(variable_pp.keys())


def preprocess(df: pd.DataFrame):
    for variable in get_variables():
        df = variable_pp[variable](df)
    df = total_drug_use(df)
    df = df.rename(columns=variable_names)
    return df
