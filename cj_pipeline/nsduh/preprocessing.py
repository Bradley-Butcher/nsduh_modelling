import pandas as pd
from functools import reduce
from cj_pipeline.config import logger
import numpy as np

# from cj_pipeline.nsduh.load import load_nsduh  # TODO: creates circular import dependency with load.py


def process_catag(df, name):
  df[name] = df[name].astype("int")
  df = df.dropna(subset=[name], axis=0)
  return df


# def process_newrace2(df, name):
#     df[name] = df[name].astype("int")
#
#     def _process(row):
#       if row[name] == 1:
#         return 'White'
#       if row[name] == 2:
#         return 'Black'
#       return 'Other'
#
#     df[name] = df.apply(_process, axis=1)
#     df = df.dropna(subset=[name], axis=0)
#     return df


def process_eduhighcat(df, name):
    df[name] = df[name].astype("int")
    d = {
        1: "<High School",
        2: "High School",
        3: "Some College",
        4: "College Graduate",
        5: "12-17 years old",
    }
    df[name] = df[name].map(d)
    df = df.dropna(subset=[name], axis=0)
    return df


def process_irsex(df, name):
    df[name] = df[name].astype("int")
    d = {
        1: "Male",
        2: "Female"
    }
    df[name] = df[name].map(d)
    df = df.dropna(subset=[name], axis=0)
    return df


def process_integer(df, name):
  df[name] = df[name].astype(pd.Int32Dtype())  # 'int' doesn't handle NaNs
  return df


def _nan_value(entry):
  return entry if pd.notna(entry) else None


def add_race(df):
  def _process(row):
    col = 'NEWRACE2'
    if _nan_value(row[col]) == 1:
      return 'White'
    if _nan_value(row[col]) == 2:
      return 'Black'

    col = 'IRRACE'
    if _nan_value(row[col]) == 3:
      return 'Black'
    if _nan_value(row[col]) == 4:
      return 'White'

    return None

  df['offender_race'] = df.apply(_process, axis=1)
  df = df.dropna(subset=['offender_race'], axis=0)
  df = df.drop(columns=['NEWRACE2', 'IRRACE'], errors='ignore')
  return df


def add_age(df):
  def _process(row):
    if _nan_value(row['CATAG3']) == 1:
      return '< 18'
    if _nan_value(row['CATAG3']) in {2, 3}:
      return '18-34'
    if _nan_value(row['CATAG3']) in {4, 5}:
      return '> 34'
    # if row['CATAG7'] in {1, 2, 3}:
    #   return '< 18'
    # if row['CATAG7'] in {4, 5, 6}:
    #   return '18-30'
    # if row['CATAG6'] in {4, 5, 6}:
    #   return '> 30'
    return None

  df['offender_age'] = df.apply(_process, axis=1)
  df = df.drop(columns=['CATAG3', 'CATAG6', 'CATAG7'], errors='ignore')
  return df


def add_dui(df):
  def _dui(df):
    grp0 = {'DRVINALCO2', 'DRVINMARJ2', 'DRVINDRG', 'DRVINDROTMJ', 'DRVINALDRG'}
    grp1 = {'DRVALDR', 'DRVAONLY', 'DRVDONLY'}
    grp2 = {'DRUNKDRV', 'DRDRVUN', 'DRIVEAL', 'DRIVEDR'}
    union = grp0.union(grp1, grp2)

    def _process(row):
      if any(_nan_value(row[col]) == 1 for col in grp0.union(grp1)):
        return True
      if any(_nan_value(row[col]) in {1, 3} for col in grp2):
        return True
      if any(_nan_value(row[col]) == 0 for col in grp0):
        return False
      if any(_nan_value(row[col]) in {2, 81, 91, 99} for col in grp1):
        return False
      if any(_nan_value(row[col]) == 2 for col in grp2):
        return False
      return None

    df['dui'] = df.apply(_process, axis=1)
    df = df.drop(columns=list(union), errors='ignore')
    return df

  def _dui_arrest(df):
    def _process(row):
      if _nan_value(row['BKDRVINF']) in {1, 3}:
        return True
      if _nan_value(row['BKOTHOFF']) == 10:
        return True
      if _nan_value(row['BKDRVINF']) in {2, 89, 99}:
        return False
      return None

    df['dui_arrests'] = df.apply(_process, axis=1)
    df = df.drop(columns=['BKDRVINF', 'BKOTHOFF'], errors='ignore')
    return df

  df = _dui(df)
  df = _dui_arrest(df)
  return df


def add_drugs(df):
  def _drugs_arrest(df):
    def _process(row):
      if _nan_value(row['BKDRUG']) in {1, 3}:
        return True
      if _nan_value(row['BKDRUG']) in {2, 4, 99}:
        return False
      return None

    df['drugs_arrest'] = df.apply(_process, axis=1)
    df = df.drop(columns='BKDRUG', errors='ignore')
    return df

  def _drugs_sold(df):
    grp0 = {'YEYSELL', 'SNYSELL'}
    grp1 = {'SOLDDRUG'}
    union = grp0.union(grp1)

    def _process(row):
      if any(_nan_value(row[col]) in {2, 3, 4, 5} for col in grp0):
        return True
      if any(_nan_value(row[col]) in {1, 3} for col in grp1):
        return True
      if any(_nan_value(row[col]) == 1 for col in grp0):
        return False
      if any(_nan_value(row[col]) == 2 for col in grp1):
        return False
      return None

    df['drugs_sold'] = df.apply(_process, axis=1)
    df = df.drop(columns=list(union), errors='ignore')
    return df

  def _drugs_use(df):
    grp0 = {'MRJMON', 'COCMON', 'CRKMON', 'HERMON', 'HALLUCMON', 'LSDMON',
           'PCPMON', 'ECSTMOMON', 'DAMTFXMON', 'KETMINMON', 'SALVIAMON',
           'INHALMON', 'METHAMMON'}
    grp1 = {'HALLREC'}
    union = grp0.union(grp1)

    def _process(row):
      if any(_nan_value(row[col]) == 1 for col in grp0):
        return True
      if any(_nan_value(row[col]) in {1, 7} for col in grp1):
        return True
      if any(_nan_value(row[col]) == 0 for col in grp0):
        return False
      return None
    df['drugs_use'] = df.apply(_process, axis=1)
    df = df.drop(columns=list(union), errors='ignore')
    return df

  df = _drugs_arrest(df)
  df = _drugs_sold(df)
  df = _drugs_use(df)
  return df


variable_pp = {  # checking presence in 1992
    "CATAG3": process_catag,
    "CATAG6": process_catag, # not present
    "CATAG7": process_catag, # not present
    # "NEWRACE2": process_newrace2, # not present
    "NEWRACE2": process_integer,
    "IRRACE": process_integer,
    "IRSEX": process_irsex, # present
    "BKDRUG": process_integer,  # present
    "BKDRVINF": process_integer, # present
    "DRVINALCO2": process_integer, # not present. alt: DRUNKDRV
    "DRVINMARJ2": process_integer, # not present
    "DRVINDRG": process_integer, # not present. potential alt: DFDRVIJN
    "DRVINDROTMJ": process_integer, # not present
    "DRVINALDRG": process_integer, # not present. potential alt: DRDRVUN
    "DRVALDR": process_integer, # not present
    "DRVAONLY": process_integer, # not present
    "DRVDONLY": process_integer, # not present
    "YEYSELL": process_integer, # not present. potential alt: SOLDDRUG
    "SNYSELL": process_integer,  # not present. potential alt: SOLDDRUG
    "MRJMON": process_integer, # present
    "COCMON": process_integer, # present
    "CRKMON": process_integer, # present
    "HERMON": process_integer, # present
    "HALLUCMON": process_integer, # not present
    "LSDMON": process_integer, # not present
    "PCPMON": process_integer, # present
    "ECSTMOMON": process_integer, # not present
    "DAMTFXMON": process_integer, # not present
    "KETMINMON": process_integer, # not present
    "SALVIAMON": process_integer, # not present
    "INHALMON": process_integer, # not present
    "METHAMMON": process_integer, # not present
    "YEAR": process_integer, # present

    "DRUNKDRV": process_integer,
    "DRDRVUN": process_integer,
    "DRIVEAL": process_integer,
    "DRIVEDR": process_integer,
    "BKOTHOFF": process_integer,
    "SOLDDRUG": process_integer,
    "HALLREC": process_integer,
}
variable_names = {
    # "NEWRACE2": "offender_race",
    "EDUHIGHCAT": "Education",
    "IRSEX": "offender_sex",
}


def get_variables():
    return set(variable_pp.keys())


def preprocess(df: pd.DataFrame, min_year: int = -1, max_year: int = np.inf):
    # df = load_nsduh()  # TODO: creates circular import dependency with load.py
    df = df[(df['YEAR'] >= min_year) & (df['YEAR'] <= max_year)]
    logger.info(f"Preprocessing data")
    for variable in get_variables():
        if variable in df.columns:
            df = variable_pp[variable](df, name=variable)
        else:
            df[variable] = None  # set all values as missing
    logger.info(f"Preprocessing age")
    df = add_age(df)
    logger.info(f"Preprocessing race")
    df = add_race(df)
    logger.info(f"Preprocessing DUI")
    df = add_dui(df)
    logger.info(f"Preprocessing drugs")
    df = add_drugs(df)
    df = df.rename(columns=variable_names)

    # TODO: split into general preprocessing (above) and date specific (below)
    #  to save computation? (the general can be already applied at load_nsduh)

    logger.info(f"Computing arrest rates")
    dfs, groups = [], ["offender_race", "offender_age", "offender_sex"]
    dfs.append(df.groupby(groups).size().to_frame('count').reset_index())
    dfs.append(df.groupby(groups).apply(
      lambda g: g['dui_arrests'].sum() / g['dui'].sum()
    ).to_frame('dui_arrest_rate').reset_index())
    dfs.append(df.groupby(groups).apply(
      lambda g: (g['drugs_arrest'] * g['drugs_use']).sum() / g['drugs_use'].sum()
    ).to_frame('drugs_user_arrest_rate').reset_index())
    dfs.append(df.groupby(groups).apply(
      lambda g: (g['drugs_arrest'] * g['drugs_sold']).sum() / g['drugs_sold'].sum()
    ).to_frame('drugs_seller_arrest_rate').reset_index())
    dfs.append(df.groupby(groups).apply(
      lambda g: g['drugs_arrest'].sum() / ((g['drugs_use'] + g['drugs_sold']) > 0).sum()
    ).to_frame('drugs_any_arrest_rate').reset_index())
    df = reduce(lambda df0, df1: pd.merge(df0, df1, on=groups), dfs)
    return df