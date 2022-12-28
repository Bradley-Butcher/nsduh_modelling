"""Script to calculate OGRS3 coefficients for a given row of data."""

def get_ogrs3_conviction_coef(row):
    if row['prior.conviction'] == 0 and row['conviction'] == 0 and row['prior.n.arrest.not.dismissed'] == 0:
        return 0
    elif row['prior.conviction'] == 0 and row['conviction'] == 0 and row['prior.n.arrest.not.dismissed'] == 1:
        return 0.083100501
    elif row['prior.conviction'] == 0 and row['conviction'] == 1 and row['prior.n.arrest.not.dismissed'] == 0:
        return 0.126142106
    elif row['conviction'] == 1 and row['prior.conviction'] > 0:
        return 0.463062792
    else:
        return 0.34859587

def ogrs3_gender_coef(row):
    if row['def.gender'] == 'Male':
        if row['def.age.years'] >= 10 and row['def.age.years'] < 12:
            return 0
        elif row['def.age.years'] >= 12 and row['def.age.years'] < 14:
            return 0.083922902
        elif row['def.age.years'] >= 14 and row['def.age.years'] < 16:
            return 0.075775765
        elif row['def.age.years'] >= 16 and row['def.age.years'] < 18:
            return -0.061594199
        elif row['def.age.years'] >= 18 and row['def.age.years'] < 21:
            return -0.625103618
        elif row['def.age.years'] >= 21 and row['def.age.years'] < 25:
            return -1.051515067
        elif row['def.age.years'] >= 25 and row['def.age.years'] < 30:
            return -1.166679288
        elif row['def.age.years'] >= 30 and row['def.age.years'] < 35:
            return -1.325976554
        elif row['def.age.years'] >= 35 and row['def.age.years'] < 40:
            return -1.368045933
        elif row['def.age.years'] >= 40 and row['def.age.years'] < 50:
            return -1.499690953
        else:
            return -2.025261458
    elif row['def.gender'] == 'Female':
        if row['def.age.years'] >= 10 and row['def.age.years'] < 12:
            return -0.785038489
        elif row['def.age.years'] >= 12 and row['def.age.years'] < 14:
            return -0.613852078
        elif row['def.age.years'] >= 14 and row['def.age.years'] < 16:
            return -0.669521331
        elif row['def.age.years'] >= 16 and row['def.age.years'] < 18:
            return -0.959179629
        elif row['def.age.years'] >= 18 and row['def.age.years'] < 21:
            return -0.897480934
        elif row['def.age.years'] >= 21 and row['def.age.years'] < 25:
            return -1.028488454
        elif row['def.age.years'] >= 25 and row['def.age.years'] < 30:
            return -1.052777806
        elif row['def.age.years'] >= 30 and row['def.age.years'] < 35:
            return -1.129127959
        elif row['def.age.years'] >= 35 and row['def.age.years'] < 40:
            return -1.42187494
        elif row['def.age.years'] >= 40 and row['def.age.years'] < 50:
            return -1.524652221
        else:
            return -2.44983716
    else:
        return 0
