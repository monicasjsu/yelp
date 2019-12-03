import pandas as pd
import re

from db.mysql import Engine

# df = pd.read_csv('yelp_cleaned.csv')
db_conn = Engine.get_db_conn()
df = pd.read_sql('yelp_cleaned', db_conn)

df.head()
print(df.columns)

# Lets drop the columns which are of no use to us.
cols_drop = ['address', 'business_id']
df.drop(cols_drop, axis=1, inplace=True)
list(df.isnull().sum())

# Now lets remove the rows where the missing values are more than 60%

# Before that, lets check the features we are planning to remove and note down any features which are important.
# Then lets try to fill in the missing values.

null_features = [col for col in df.columns if df[col].isnull().sum() / df.shape[0] > 0.60]
print("These features has more than 60% missing values\n" + str(null_features))

# Looks like happy hour is the only important feature missing.

# So lets try to impute the values using the following logic.
# Assumption: Restaurants having full_bar may have the happy hour.

for record in df.iterrows():
    if record[1].isnull()['HappyHour']:
        if record[1]['Alcohol'] == 'full bar':
            df.set_value(record[0], 'HappyHour', 'True')
        elif record[1]['Alcohol'] == 'beer and wine':
            df.set_value(record[0], 'HappyHour', 'False')
        elif record[1]['Alcohol'] == 'None':
            df.set_value(record[0], 'HappyHour', 'False')

# Now lets check the null features to be removed again
null_features = [col for col in df.columns if df[col].isnull().sum() / df.shape[0] > 0.60]
df.drop(null_features, axis=1, inplace=True)

val_counts = df['city'].value_counts()
print(val_counts)

df_vegas = df[df['city'].str.contains('vegas', flags=re.IGNORECASE, regex=True)]
df_vegas.to_sql('yelp_vegas', db_conn, index=False)
# df_vegas.to_csv('yelp_vegas.csv')

df_toronto = df[df['city'].str.contains('toronto', flags=re.IGNORECASE, regex=True)]
df_toronto.to_sql('yelp_toronto', db_conn, index=False)
# df_vegas.to_csv('yelp_toronto.csv')

df_phoenix = df[df['city'].str.contains('toronto', flags=re.IGNORECASE, regex=True)]
df_phoenix.to_sql('yelp_phoenix', db_conn, index=False)
# df_vegas.to_csv('yelp_phoenix.csv')

