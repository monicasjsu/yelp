import pandas as pd
from db.mysql import Engine

# If you want to load the dataset from local, uncomment this code.
# df = pd.read_csv('yelp_preprocessed.csv')
db_conn = Engine.get_db_conn()
df = pd.read_sql('yelp_raw', db_conn)

df = df[df['categories'].str.contains('Restaurant', na=False)]
print(df.head())

# Remove attributes with null/missing values more than 95%
null_features = [col for col in df.columns if df[col].isnull().sum() / df.shape[0] > 0.95]
print("These features will be removed as there are more than 95% missing values\n" + str(null_features))
df.drop(null_features, axis=1, inplace=True)

df = df[pd.notnull(df['categories'])]
print(df['categories'].isnull().sum())

# Split the categories column which has a list of words(comma separated) into columns
categories = df['categories'].str.split(',').tolist()
max_cat = 0
for row_cat in categories:
    if len(row_cat) > max_cat:
        max_cat = len(row_cat)
cat_columns = ['category_' + str(i) for i in range(0, max_cat)]
df_categories = pd.DataFrame(categories, columns=cat_columns)

# Create dummies (one hot encoding) for all these category columns.
df_categories_dummy = pd.get_dummies(df_categories)
df_categories_dummy.columns = df_categories_dummy.columns.str.replace('category_[0-9]{1,2}_', '')
df_categories_dummy.columns = df_categories_dummy.columns.str.strip()

# Perform a sum operation on duplicate columns resulted from one hot encoding.
df_categories_sum = df_categories_dummy.groupby(df_categories_dummy.columns, axis=1, sort=True, squeeze=True).sum()

# Delete the unnecessary data frames to save memory
del df_categories
del df_categories_dummy

# Read the restaurant_related_words.csv file and filter out columns that exist in the file.
# These are the only restaurant words we need to work on.
df_restaurant_valid_words = pd.read_csv('restaurant_related_words.csv')
restaurant_valid_words = df_restaurant_valid_words.keys().tolist()
df_categories_sum.columns = df_categories_sum.columns.str.strip()
# Use the set intersection to filter out the columns with labels that are not in the file.
df_valid_categories = df_categories_sum[set(restaurant_valid_words).intersection(df_categories_sum.columns)]
# Drop the extra Karaoke column as that was extracted due to one hot encoding.
df_valid_categories.drop('Karaoke', axis=1, inplace=True)


del df_categories_sum

df.reset_index(inplace=True)
df = pd.concat([df, df_valid_categories], axis=1)
df.drop(['categories', 'index'], axis=1, inplace=True)
df.to_sql('yelp_cleaned', con=db_conn, index=False)

# Uncomment if you want to export the Data frame to local.
# df.to_csv('yelp_cleaned.csv')

