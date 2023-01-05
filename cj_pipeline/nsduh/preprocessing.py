import pandas as pd
from functools import reduce
from cj_pipeline.config import logger

# def process_bkdrug(df: pd.DataFrame, name: str):
#     df[name] = df[name].astype("int")
#     d = {
#         1: 1,
#         2: 0,
#         3: 1,
#         99: 0
#     }
#     df[name] = df[name].map(d)
#     df = df.dropna(subset=[name], axis=0)
#     return df


# def process_drug(df, name):
#     df[name] = df[name].astype("int")
#     df.loc[df[name] > 365, name] = 0
#     return df


def process_catag(df, name):
  df[name] = df[name].astype("int")
  df = df.dropna(subset=[name], axis=0)
  return df


def process_newrace2(df, name):
    df[name] = df[name].astype("int")

    def _process(row):
      if row[name] == 1:
        return 'White'
      if row[name] == 2:
        return 'Black'
      return 'Other'

    df[name] = df.apply(_process, axis=1)
    df = df.dropna(subset=[name], axis=0)

    return df


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


# def process_snysell(df, name):
#     df[name] = df[name].astype("int")
#     d = {
#         1: 0,
#         2: 1,
#         3: 2,
#         4: 3,
#         5: 4,
#         99: 0
#     }
#     df[name] = df[name].map(d)
#     df = df.dropna(subset=[name], axis=0)
#     return df


def process_integer(df, name):
  df[name] = df[name].astype(pd.Int32Dtype())  # 'int' doesn't handle NaNs
  return df


def add_age(df):
  def _process(row):
    if row['CATAG7'] in {1, 2, 3}:
      return '< 18'
    if row['CATAG7'] in {4, 5, 6}:
      return '18-30'
    if row['CATAG6'] in {4, 5, 6}:
      return '> 30'
    return None

  df['offender_age'] = df.apply(_process, axis=1)
  df = df.drop(columns=['CATAG6', 'CATAG7'])
  return df


def add_dui(df):
  def _dui(df):
    base = {'DRVINALCO2', 'DRVINMARJ2', 'DRVINDRG', 'DRVINDROTMJ', 'DRVINALDRG'}
    additional = {'DRVALDR', 'DRVAONLY', 'DRVDONLY'}
    union = base.union(additional)

    def _process(row):
      if any(pd.notna(row[col]) and row[col] == 1 for col in union):
        return True
      if any(pd.notna(row[col]) and row[col] == 0 for col in base):
        return False
      if any(pd.notna(row[col]) and row[col] in {2, 81, 91, 99} for col in additional):
        return False
      # if any(pd.notna(row[col]) and row[col] in {81, 99}
      #        for col in base.difference({'DRVINMARJ2', 'DRVINDRG'})):
      #   return False
      return None

    df['dui'] = df.apply(_process, axis=1)
    df = df.drop(columns=list(union))
    return df

  def _dui_arrest(df):
    def _process(row):
      if row['BKDRVINF'] in {1, 3}:
        return True
      if row['BKDRVINF'] in {2, 89, 99}:
        return False
      return None
    df['dui_arrests'] = df.apply(_process, axis=1)
    df = df.drop(columns='BKDRVINF')
    return df

  df = _dui(df)
  df = _dui_arrest(df)
  return df


def add_drugs(df):
  def _drugs_arrest(df):
    def _process(row):
      if row['BKDRUG'] in {1, 3}:
        return True
      if row['BKDRUG'] in {2, 4, 99}:
        return False
      return None

    df['drugs_arrest'] = df.apply(_process, axis=1)
    df = df.drop(columns='BKDRUG')
    return df

  def _drugs_sold(df):
    def _process(row):
      if row['YEYSELL'] in {2, 3, 4, 5} or row['SNYSELL'] in {2, 3, 4, 5}:
        return True
      if row['YEYSELL'] == 1 or row['SNYSELL'] == 1:
        return False
      return None

    df['drugs_sold'] = df.apply(_process, axis=1)
    df = df.drop(columns=['YEYSELL', 'SNYSELL'])
    return df

  def _drugs_use(df):
    mon = {'MRJMON', 'COCMON', 'CRKMON', 'HERMON', 'HALLUCMON', 'LSDMON',
           'PCPMON', 'ECSTMOMON', 'DAMTFXMON', 'KETMINMON', 'SALVIAMON',
           'INHALMON', 'METHAMMON'}

    def _process(row):
      if any(pd.notna(row[col]) and row[col] == 1 for col in mon):
        return True
      if any(pd.notna(row[col]) and row[col] == 0 for col in mon):
        return False
      return None

    df['drugs_use'] = df.apply(_process, axis=1)
    df = df.drop(columns=list(mon))
    return df

  df = _drugs_arrest(df)
  df = _drugs_sold(df)
  df = _drugs_use(df)
  return df


variable_pp = {
    # "BKDRUG": process_bkdrug,
    # "IRMJFY": process_drug,
    # "IRCOCFY": process_drug,
    # "IRHERFY": process_drug,
    # "IRCRKFY": process_drug,
    "CATAG6": process_catag,
    "CATAG7": process_catag,
    "NEWRACE2": process_newrace2,
    # "EDUHIGHCAT": process_eduhighcat,
    "IRSEX": process_irsex,
    # "SNYSELL": process_snysell,

    "BKDRUG": process_integer,  # alt above transform
    "BKDRVINF": process_integer,
    "DRVINALCO2": process_integer,
    "DRVINMARJ2": process_integer,
    "DRVINDRG": process_integer,
    "DRVINDROTMJ": process_integer,
    "DRVINALDRG": process_integer,
    "DRVALDR": process_integer,
    "DRVAONLY": process_integer,
    "DRVDONLY": process_integer,
    # "DRIVALD2": process_integer,
    # "DRVUNDALD": process_integer,

    "YEYSELL": process_integer,
    "SNYSELL": process_integer,  # alt above transform
    "MRJMON": process_integer,
    "COCMON": process_integer,
    "CRKMON": process_integer,
    "HERMON": process_integer,
    "HALLUCMON": process_integer,
    "LSDMON": process_integer,
    "PCPMON": process_integer,
    "ECSTMOMON": process_integer,
    "DAMTFXMON": process_integer,
    "KETMINMON": process_integer,
    "SALVIAMON": process_integer,
    "INHALMON": process_integer,
    "METHAMMON": process_integer,
}
# drug_vars = {"IRMJFY", "IRCOCFY", "IRHERFY", "IRCRKFY"}
variable_names = {
    # "BKDRUG": "Arrested",
    # "IRMJFY": "Marijuana Past Year",
    # "IRCOCFY": "Cocaine Past Year",
    # "IRHERFY": "Heroin Past Year",
    # "IRCRKFY": "Crack Past Year",
    "NEWRACE2": "offender_race",
    "EDUHIGHCAT": "Education",
    "IRSEX": "offender_sex",
    # "SNYSELL": "Sold Drugs Past Year"
}


# def total_drug_use(df):
#     df['Total Drug Use'] = df[drug_vars].sum(axis=1)
#     return df


def get_variables():
    return set(variable_pp.keys())


def preprocess(df: pd.DataFrame):
    logger.info(f"Preprocessing data")
    for variable in get_variables():
        df = variable_pp[variable](df, name=variable)
    logger.info(f"Preprocessing age")
    df = add_age(df)
    logger.info(f"Preprocessing DUI")
    df = add_dui(df)
    logger.info(f"Preprocessing drugs")
    df = add_drugs(df)
    # df = total_drug_use(df)
    df = df.rename(columns=variable_names)

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
    # df = df.groupby(groups).agg(
    #     {
    #         "Marijuana Past Year": "sum",
    #         "Cocaine Past Year": "sum",
    #         "Heroin Past Year": "sum",
    #         "Crack Past Year": "sum",
    #         "Total Drug Use": "sum",
    #         "Sold Drugs Past Year": "sum",
    #         "Arrested": "sum"
    #     }
    # )
    # df = df.merge(counts, on=groups)
    return df
