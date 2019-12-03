import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

from db.mysql import Engine

db_conn = Engine.get_db_conn()
df = pd.read_sql('yelp_vegas', db_conn)
print(df.columns)

# Find missing values
print(df.shape)
print(list(df.isnull().sum()))

# Delete samples with missing values more than or equal to 8 columns
df.dropna(thresh=df.shape[1] - 8, inplace=True)

print(df.shape)
print(df.isnull().sum())

df.replace('', np.nan, inplace=True)

# Changes the data type of postal_code to int and plot them
df_postal = df['postal_code'].dropna().apply(np.int64)
df_postal.value_counts().plot(kind='bar', figsize=(25, 5))

# Lets view Restaurant Attire value_counts()
print(df['RestaurantsAttire'].value_counts())
# As casual attire has most of the value counts as casual attire.
# It can be removed.

columns_drop = ['latitude', 'longitude', 'name', 'postal_code',
                'state', 'BikeParking', 'is_open', 'RestaurantsAttire']
df.drop(columns_drop, axis=1, inplace=True)

# Lets replace All True/False to 1/0 s
df.replace(['TRUE', 'true', 'True', True], 1, inplace=True)
df.replace(['FALSE', 'false', 'False', False], 0, inplace=True)


def clean_alcohol(alcohol_str):
    if isinstance(alcohol_str, str):
        if 'none' in alcohol_str:
            return np.int8(0)
        else:
            return np.int8(1)


print(df.isnull().sum().value_counts())

df['Alcohol'] = df.Alcohol.apply(clean_alcohol)


# print(df['Alcohol'].head())


def clean_noise_level(level):
    if isinstance(level, str):
        if 'quiet' in level:
            return np.int8(1)
        elif 'average' in level:
            return np.int8(2)
        elif 'loud' in level:
            return np.int8(3)
        elif 'very_loud' in level:
            return np.int8(4)
        else:
            return np.int8(0)


df['NoiseLevel'] = df.NoiseLevel.apply(clean_noise_level)


def clean_wifi(wifi):
    if isinstance(wifi, str):
        if 'None' in wifi:
            return np.int8(0)
        elif 'no' in wifi:
            return np.int8(0)
        else:
            return np.int8(1)


df['WiFi'] = df.WiFi.apply(clean_wifi)

print(df['BusinessAcceptsCreditCards'].value_counts(normalize=True))

for column in df.columns:
    print(df[column].value_counts(normalize=True))

# some features have float values. convert them to int before applying filter to remove repeated values.
# float_cols = ['trendy', 'upscale', 'breakfast', 'brunch', 'dessert', 'dinner', 'latenight']
#
# df[float_cols].replace(['1.0', 1.0], 1, inplace=True)
# df[float_cols].replace(['0.0', 0.0], 0, inplace=True)


# Drop cols which have zeros greater than 95%
drop_cols = [col for col in df.columns if df[col].value_counts(normalize=True).values[0] > 0.95]

# As more than 99% of restaurants accept Credit cards, the BusinessAcceptsCreditCards feature can be removed.

# Preserve the columns which looks important fo insights
cols_to_preserve = ['classy', 'divey', 'hipster', 'intimate',
                    'romantic', 'touristy', 'upscale', 'Thai', 'Mongolian', 'Korean', 'Vegetarian',
                    'Buffets', 'Mediterranean', 'Vegan', 'French', 'Indian']

drop_cols_final = [x for x in drop_cols if x not in cols_to_preserve]

df.drop(drop_cols_final, axis=1, inplace=True)
print(drop_cols_final)
print(df.shape)

# Drop samples which have price range missing values, as we consider it as im important feature.
# same for other features, as they have a very minimum number of missing values.
df.dropna(subset=['RestaurantsPriceRange2'], how='any', inplace=True)
df.dropna(subset=['RestaurantsGoodForGroups'], how='any', inplace=True)
df.dropna(subset=['GoodForKids'], how='any', inplace=True)
df.dropna(subset=['RestaurantsTakeOut'], how='any', inplace=True)
df.dropna(subset=['upscale'], how='any', inplace=True)

# Remove all rows with String 'None'
df = df[~df.isin(['None', 'none', None])]

# Fill missing values.
# If good for kids = 1, then Alcohol = 0 and vice versa
for index, row in df.iterrows():
    if row.isnull()['Alcohol']:
        if row['GoodForKids'] == 1:
            df.set_value(index, 'Alcohol', 0)
        else:
            df.set_value(index, 'Alcohol', 1)

# Fill Restaurant Delivery by using filling it using the most occurring element of the column
df['RestaurantsDelivery'].fillna(df['RestaurantsDelivery'].mode()[0], inplace=True)

# Assumption: If Restaurant provides Restaurant Delivery, then they can also caters
for index, row in df.iterrows():
    if row.isnull()['Caters']:
        if row['RestaurantsDelivery'] == 1:
            df.set_value(index, 'Caters', 1)
        else:
            df.set_value(index, 'Caters', 0)

# Fill HasTV by using filling it using the most occurring element of the column
df['HasTV'].fillna(df['HasTV'].mode()[0], inplace=True)

for index, row in df.iterrows():
    if row.isnull()['NoiseLevel']:
        if row['HasTV'] == 1:
            df.set_value(index, 'NoiseLevel', 3)

# Fill the remaining NoiseLevel by using filling it using the most occurring element of the column
df['NoiseLevel'].fillna(df['NoiseLevel'].mode()[0], inplace=True)

# Fill OutdoorSeating by using filling it using the most occurring element of the column
df['OutdoorSeating'].fillna(df['OutdoorSeating'].mode()[0], inplace=True)

# Fill RestaurantsReservations by using filling it using the most occurring element of the column
df['RestaurantsReservations'].fillna(df['RestaurantsReservations'].mode()[0], inplace=True)

for column in df.columns:
    print(df[column].value_counts(normalize=True))

# Change the type of RestaurantsPriceRange2 from str to int
df = df.astype({'RestaurantsPriceRange2': 'float'})

# Assumption: If Restaurant is pricey, it offers wifi
for index, row in df.iterrows():
    if row.isnull()['WiFi']:
        if int(row['RestaurantsPriceRange2']) >= 2:
            df.set_value(index, 'WiFi', 1)
        else:
            df.set_value(index, 'WiFi', 0)

# Fill remaining parking by using filling it using the most occurring element of the column
df['WiFi'].fillna(df['WiFi'].mode()[0], inplace=True)

# Merge lot and garage features as part od Dimensionality Reduction
for index, row in df.iterrows():
    if row['garage'] == 1 or row['lot'] == 1:
        df.set_value(index, 'garage', 1)
    else:
        df.set_value(index, 'garage', 0)

df.drop('lot', axis=1, inplace=True)
df.rename(columns={'garage': 'parking'}, inplace=True)

# Assumption: If Restaurant is pricey, it offers parking too
for index, row in df.iterrows():
    if row.isnull()['parking']:
        if row['RestaurantsPriceRange2'] >= 2:
            df.set_value(index, 'parking', 1)

# Fill remaining parking by using filling it using the most occurring element of the column
df['parking'].fillna(df['parking'].mode()[0], inplace=True)

# Assumption: If Restaurant is too pricey, it also offers valet too
for index, row in df.iterrows():
    if row.isnull()['valet']:
        if row['RestaurantsPriceRange2'] >= 3:
            df.set_value(index, 'valet', 1)

# Fill remaining valet by using filling it using the most occurring element of the column
df['valet'].fillna(df['valet'].mode()[0], inplace=True)

# Assumption: If Restaurant is less pricey, it can only offer street parking
for index, row in df.iterrows():
    if row.isnull()['street']:
        if row['RestaurantsPriceRange2'] == 1:
            df.set_value(index, 'street', 1)

# Fill remaining parking by using filling it using the most occurring element of the column
df['street'].fillna(df['street'].mode()[0], inplace=True)

# Fill breakfast, brunch, dessert, dinner, and latenight by using filling it using the most occurring element of columns
df['breakfast'].fillna(df['breakfast'].mode()[0], inplace=True)
df['brunch'].fillna(df['brunch'].mode()[0], inplace=True)
df['dessert'].fillna(df['dessert'].mode()[0], inplace=True)
df['dinner'].fillna(df['dinner'].mode()[0], inplace=True)
df['latenight'].fillna(df['latenight'].mode()[0], inplace=True)
df['lunch'].fillna(df['lunch'].mode()[0], inplace=True)

df.dropna(subset=['RestaurantsPriceRange2'], how='any', inplace=True)
df.dropna(subset=['GoodForKids'], how='any', inplace=True)

for column in df.columns:
    print(column)
    print(df[column].isnull().sum())
    print("=========================")

df['stars'] = df.stars.astype(int)
df['Caters'] = df.Caters.astype(int)
df['GoodForKids'] = df.GoodForKids.astype(int)

# Convert stars from float to int by multiplying it by 10
# df['stars'] = df['stars'].apply(lambda x: x * 10)

# Detect outliers in review counts
plt.figure()
sns.boxplot(x=df['review_count'])
plt.show()

q75, q25 = np.percentile(df['review_count'], [75, 25])
IQR = q75 - q25

# Lets remove the outliers with range > 3 * IQR
df = df[~((df['review_count'] > (q75 + 1.5 * IQR)) | (df['review_count'] < (q25 - 1.5 * IQR)))]
# Lets check the shape of the new DF after removing outliers.
print(df.shape)
# Now lets plot the Box plot again to see the distribution
plt.figure()
sns.boxplot(x=df['review_count'])
plt.show()

# Normalize the stars and review count and generate a new features (60% weight to stars and 40% to reviews)
min_max_scaler = MinMaxScaler()
stars_reviews = min_max_scaler.fit_transform(df[['stars', 'review_count']])
df_stars_reviews = pd.DataFrame(stars_reviews, columns=['stars_norm', 'review_norm'])
df.reset_index(inplace=True, drop=True)
df = pd.concat([df, df_stars_reviews], axis=1)

# Now lets generate a new feature called score
df['score'] = (df['review_norm'] * 40 + df['stars_norm'] * 60)
# Now lets check the score feature min, max and mean stats
print(df['score'].describe())

# Convert score to the range of 1-5
score_mean = df['score'].mean()
score_std = df['score'].std()

df['score'] = np.where(df['score'].between(0, score_mean - score_std), 1, df['score'])
df['score'] = np.where(df['score'].between(score_mean - score_std, score_mean), 2, df['score'])
df['score'] = np.where(df['score'].between(score_mean, score_mean + score_std), 3, df['score'])
df['score'] = np.where(df['score'].between(score_mean + score_std, score_mean + 2 * score_std), 4, df['score'])
df['score'] = np.where(df['score'].between(score_mean + 2 * score_std, 100), 5, df['score'])
df = df.astype({'score': 'int'})

df.drop(['city', 'review_norm', 'stars_norm'], axis=1, inplace=True)

# Now lets save the final data frame which has no missing values into databases.
# This will be the final version of dataset, where we run the models.
df.to_sql('yelp_vegas_final', db_conn, index=False)
