import numpy as np

def npdt_to_str(np_dt):
    date = np.datetime_as_string(np_dt, unit='D')
    days_left = round((np_dt - np.datetime64('now'))/ np.timedelta64(1, 'D'), 1)
    return f"{date} ({days_left} days)"


def get_default_row(df, date):

    temp_df = df[
        (df['instrument_name'].apply(lambda x : x[-1:] == 'C'))
        & (df['expiration_time'].apply(lambda x : int(x.timestamp()) == date.astype('datetime64[s]').astype('int')))
    ].sort_values('strike', ascending=True)    

    return temp_df.iloc[int(len(temp_df)/2), :]

