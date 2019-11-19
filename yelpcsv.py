import math
import re
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

df = pd.read_csv('yelp_vegas.csv')
print(df.columns)

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# restaurant_tokens = df['name'].str.split()
# restaurant_text = ' '.join(word[0] for word in restaurant_tokens.values)
# wordcloud = WordCloud(width=800, height=800, background_color='white') \
#     .generate(restaurant_text)
# plt.figure(figsize=(8, 8), facecolor=None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show()

# %%
# Find missing values
# print(df.columns)
print(df.shape)
print(list(df.isnull().sum()))

# Delete samples with more than or equal to 10 columns
df.dropna(thresh=df.shape[1]-8, inplace=True)

# print(df.columns)
print(df.shape)
print(df.isnull().sum())

df_postal = df['postal_code'].dropna().apply(np.int64)
df_postal.value_counts().plot(kind='bar', figsize=(25, 5))

# Lets view Restaurant Attire value_counts()
print(df['RestaurantsAttire'].value_counts())
# As casual attire has most of the value counts as casual attire.
# It can be removed.

columns_drop = ['Unnamed: 0', 'latitude', 'longitude', 'name', 'postal_code',
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

cols_to_preserve = ['BusinessAcceptsCreditCards', 'classy', 'divey', 'hipster', 'intimate',
                    'romantic', 'touristy', 'upscale', 'Thai', 'Mongolian', 'Korean', 'Vegetarian',
                    'Buffets', 'Mediterranean', 'Vegan', 'French', 'Indian']

drop_cols_final = [x for x in drop_cols if x not in cols_to_preserve]

df.drop(drop_cols_final, axis=1, inplace=True)
print(drop_cols_final)
print(df.shape)

# Drop sample which have price range missing values, as we consider it as im important feature.
# same for other features, as they have a very minimum number of missing values.
df.dropna(subset=['RestaurantsPriceRange2'], how='any', inplace=True)
df.dropna(subset=['RestaurantsGoodForGroups'], how='any', inplace=True)
df.dropna(subset=['BusinessAcceptsCreditCards'], how='any', inplace=True)
df.dropna(subset=['GoodForKids'], how='any', inplace=True)
df.dropna(subset=['RestaurantsTakeOut'], how='any', inplace=True)
df.dropna(subset=['upscale'], how='any', inplace=True)

# Todo: Find out the possibility of bringing in Happy hour feature

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

# Fill HasTV by using filling it using the most occurring element of the column
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
# Convert stars from float to int by multiplying it by 10
# df['stars'] = df['stars'].apply(lambda x: x * 10)

df.drop(['city', 'review_count'], axis=1, inplace=True)
# df.drop(['breakfast', 'brunch', 'dessert', 'dinner',
#          'latenight', 'lunch', 'Food', 'Vegan', 'Bars',
#          'Mongolian', 'French', 'WiFi'], axis=1, inplace=True)

y = df['stars']
X = df.drop('stars', axis=1)

print(df.shape)
print(df.columns)


df.to_csv('temp.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100, stratify=df['stars'])

# ******************* KNN ************************

# knn_range = range(3, 26)
#
# for k in knn_range:
#     print('Iteration: ' + str(k))
#     knn_model = KNeighborsClassifier(n_neighbors=k)
#     knn_model.fit(X_train, y_train)
#     pred = knn_model.predict(X_test)
#     print("\nAccuracy Score:")
#     print(accuracy_score(y_test, pred))
#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y_test, pred))
#     print("\nClassification Report:")
#     print(classification_report(y_test, pred))
#     print('==================================================================')

# ******************* SVM ************************

# svm_model = svm.SVC(kernel='linear')
# svm_model.fit(X_train, y_train)
# predicted1 = svm_model.predict(X_test)
# score = svm_model.score(X_test, y_test)
# print(score)

# *********************Logistic Regression **********************

# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# predicted = logreg.predict(X_test)
# print("\nAccuracy Score:")
# print(accuracy_score(y_test, predicted))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, predicted))
# print("\nClassification Report:")
# print(classification_report(y_test, predicted))

# ********************* Random Forest ************************

