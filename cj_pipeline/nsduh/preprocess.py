import tqdm
import itertools
import pandas as pd
from functools import reduce
from cj_pipeline.config import logger
from sklearn.linear_model import LinearRegression


def _sdiv(a, b):
  return a / b if b != 0 else 0


def _nan_value(entry):
  return entry if pd.notna(entry) else None


def _categorical_to_object_col(df, name):
  if df[name].dtype.name == 'category':
    df[name] = df[name].astype(object)


def process_integer(df, name):
  df[name] = df[name].astype(pd.Int32Dtype())  # 'int' doesn't handle NaNs
  return df


def process_newrace2(df, name):
  df[name].replace('NonHisp White', 1, inplace=True)
  df[name].replace('NonHisp Black/Afr Am', 2, inplace=True)
  df[name] = pd.to_numeric(df[name], errors='coerce')  # set all else to NaN
  return process_integer(df, name)


def process_catag3(df, name):
  df.replace({
    name: {
      '12-17 Years Old': 1,
      '18-25 Years Old': 2,
      '26-34 Years Old': 3,
      '35-49 Years Old': 4,
      '50 or Older': 5
    }
  }, inplace=True)
  df[name] = pd.to_numeric(df[name], errors='coerce')  # set all else to NaN
  return process_integer(df, name)


def process_dui_like(df, name):
  _categorical_to_object_col(df, name)

  df[name].replace('Yes', 1, inplace=True)
  df[name].replace(
    regex=[
      'No', 'No .*', 'LEGITIMATE SKIP', 'LEGITIMATE SKIP Logically assigned',
      'NEVER USED ALCOHOL OR DRUGS', 'NEVER USED ALCOHOL OR DRUGS Logically assigned'],
    value=2, inplace=True)
  df[name] = pd.to_numeric(df[name], errors='coerce')  # set all else to NaN
  return process_integer(df, name)


def process_drug_sell(df, name):
  _categorical_to_object_col(df, name)

  df[name].replace(
    ['1 or 2 times', '3 to 5 times', '6 to 9 times', '10 or more times'],
    2, inplace=True)
  df[name].replace(
    ['0 times', 'LEGITIMATE SKIP', 'LEGITIMATE SKIP Logically assigned'],
    1, inplace=True)
  df[name] = pd.to_numeric(df[name], errors='coerce')  # set all else to NaN
  return process_integer(df, name)


def process_bkdrug(df, name):
  _categorical_to_object_col(df, name)

  df[name].replace(['Yes', 'Yes LOGICALLY ASSIGNED'], 1, inplace=True)
  df[name].replace(['No', 'LEGITIMATE SKIP'], 2, inplace=True)
  df[name] = pd.to_numeric(df[name], errors='coerce')  # set all else to NaN
  return process_integer(df, name)


def process_bkdrvinf(df, name):
  _categorical_to_object_col(df, name)

  df[name].replace(['Yes', 'Yes LOGICALLY ASSIGNED'], 1, inplace=True)
  df[name].replace(['No', 'LEGITIMATE SKIP'], 2, inplace=True)
  df[name] = pd.to_numeric(df[name], errors='coerce')  # set all else to NaN
  return process_integer(df, name)


def process_drugmon(df, name):
  _categorical_to_object_col(df, name)

  df[name].replace(inplace=True, regex={
    'Used within the past month .*': 1, 'Did not use in the past month .*': 0
  })
  df[name] = pd.to_numeric(df[name], errors='coerce')  # set all else to NaN
  return process_integer(df, name)


def process_hallrec(df, name):
  _categorical_to_object_col(df, name)

  df[name].replace([
    'Within the past 30 days', 'Used in the past 30 days LOGICALLY ASSIGNED'],
    1, inplace=True)
  df[name].replace([
    'NEVER USED HALLUCINOGENS',
    'More than 30 days ago but within the past 12 mos',
    'Used >30 days ago but within pst 12 mos LOG ASSN',
    'More than 12 months ago'],
    0, inplace=True)
  df[name] = pd.to_numeric(df[name], errors='coerce')  # set all else to NaN
  return process_integer(df, name)


def process_irsex(df, name):
  df[name].replace(1, 'Male', inplace=True)
  df[name].replace(2, 'Female', inplace=True)
  df = df.dropna(subset=[name], axis=0)
  return df


# def process_eduhighcat(df, name):
#     df[name] = df[name].astype("int")  # likely not updated for stata
#     d = {
#         1: "<High School",
#         2: "High School",
#         3: "Some College",
#         4: "College Graduate",
#         5: "12-17 years old",
#     }
#     df[name] = df[name].map(d)
#     df = df.dropna(subset=[name], axis=0)
#     return df


variable_pp = {  # checking presence in 1992
    "CATAG3": process_catag3,
    # "CATAG6": process_integer, # not present
    # "CATAG7": process_integer, # not present
    "NEWRACE2": process_newrace2, # not present
    "IRRACE": process_integer,
    "IRSEX": process_irsex, # present
    "BKDRUG": process_bkdrug,  # present
    "BKDRVINF": process_bkdrvinf, # present
    "DRVINALCO2": process_integer, # not present. alt: DRUNKDRV
    "DRVINMARJ2": process_integer, # not present
    "DRVINDRG": process_integer, # not present. potential alt: DFDRVIJN
    "DRVINDROTMJ": process_integer, # not present
    "DRVINALDRG": process_integer, # not present. potential alt: DRDRVUN
    "DRVALDR": process_dui_like, # not present
    "DRVAONLY": process_dui_like, # not present
    "DRVDONLY": process_dui_like, # not present
    "YEYSELL": process_drug_sell, # not present. potential alt: SOLDDRUG
    "SNYSELL": process_drug_sell,  # not present. potential alt: SOLDDRUG
    "MRJMON": process_drugmon, # present
    "COCMON": process_drugmon, # present
    "CRKMON": process_drugmon, # present
    "HERMON": process_drugmon, # present
    "HALLUCMON": process_drugmon, # not present
    "LSDMON": process_drugmon, # not present
    "PCPMON": process_drugmon, # present
    "ECSTMOMON": process_drugmon, # not present
    "DAMTFXMON": process_drugmon, # not present
    "KETMINMON": process_drugmon, # not present
    "SALVIAMON": process_drugmon, # not present
    "INHALMON": process_drugmon, # not present
    "METHAMMON": process_drugmon, # not present
    "YEAR": process_integer, # present

    "DRUNKDRV": process_integer,  # process_dui_like
    "DRDRVUN": process_integer,  # process_dui_like
    "DRIVEAL": process_dui_like,
    "DRIVEDR": process_dui_like,
    "BKOTHOFF": process_integer,
    "SOLDDRUG": process_integer,
    "HALLREC": process_hallrec,
}


def get_variables():
    return set(variable_pp.keys())


def add_race(df):
  def _process(row):
    if _nan_value(row['NEWRACE2']) == 1 or _nan_value(row['IRRACE']) == 4:
      return 'White'
    if _nan_value(row['NEWRACE2']) == 2 or _nan_value(row['IRRACE']) == 3:
      return 'Black'
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
  df = df.dropna(subset=['offender_age'], axis=0)
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
      if any(_nan_value(row[col]) == 0 for col in grp0.union(grp1)):
        return False
      if any(_nan_value(row[col]) in {2, 3, 12, 91} for col in grp1):
        return False
      return None
    df['drugs_use'] = df.apply(_process, axis=1)
    df = df.drop(columns=list(union), errors='ignore')
    return df

  df = _drugs_arrest(df)
  df = _drugs_sold(df)
  df = _drugs_use(df)
  return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Preprocessing data")
    for variable in tqdm.tqdm(get_variables()):
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

    df = df.rename(columns={"EDUHIGHCAT": "education", "IRSEX": "offender_sex"})

    logger.info('Compute arrest rates')
    df = compute_arrest_rates(df)

    return df


def compute_arrest_rates(df: pd.DataFrame, eps: float = 0.0) -> pd.DataFrame:
  groups = ['offender_race', 'offender_age', 'offender_sex', 'YEAR']
  spec = {
    'dui': lambda g: _sdiv(g['dui_arrests'].sum(), g['dui'].sum()),
    'drugs_use': lambda g: _sdiv((g['drugs_arrest'] * g['drugs_use']).sum(), g['drugs_use'].sum()),
    'drugs_sell': lambda g: _sdiv((g['drugs_arrest'] * g['drugs_sold']).sum(), g['drugs_sold'].sum()),
    'drugs_any': lambda g: _sdiv(g['drugs_arrest'].sum(), ((g['drugs_use'] + g['drugs_sold']) > 0).sum())
  }
  grouped = df.groupby(groups)
  counts = grouped.size().to_frame('count').reset_index()

  # ensure we predict data for all years and groups
  all_combinations = itertools.product(*[df[c].unique() for c in groups])
  all_combinations = pd.DataFrame(all_combinations, columns=groups)
  agg = pd.merge(all_combinations, counts, how='left', on=groups)

  for var in spec:
    x_col, y_col, smooth_col = 'YEAR', f'{var}_ar', f'{var}_sar'
    data = grouped.apply(spec[var]).to_frame(y_col).reset_index()
    x_train = data[x_col].to_numpy()[:, None]
    y_train, weights = data[y_col], counts['count']
    x_test = agg[x_col].to_numpy()[:, None]

    model = LinearRegression()
    model.fit(x_train, y_train, weights)
    smoothed = model.predict(x_test).clip(min=eps)

    agg[smooth_col] = smoothed
    agg = pd.merge(agg, data, how='left', on=groups)

  return agg
