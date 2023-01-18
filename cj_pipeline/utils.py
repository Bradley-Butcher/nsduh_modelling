def subset_pd_bool(df, **kwargs):
  cond = True
  for key, val in kwargs.items():
    cond &= df[key] == val
  return cond


def subset_pd(df, **kwargs):
  return df[subset_pd_bool(df, **kwargs)]
