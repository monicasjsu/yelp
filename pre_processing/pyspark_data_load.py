import pandas as pd
import findspark

from db.mysql import Engine

findspark.init()

from pyspark.shell import spark
from pyspark.sql.functions import udf


json_df = spark.read.json('../dataset/yelp_academic_dataset_business.json')
json_df.printSchema()
df = json_df.select("*", "attributes.*")


def json_str_clean_up(x):
    if x is not None:
        return x.replace("False", "false").replace("True", "true").replace("'", '"')


json_str_clean_up_udf = udf(json_str_clean_up)

df_json_cleaned_up = df.select(
    json_str_clean_up_udf("BusinessParking").alias("parking"),
    json_str_clean_up_udf("Ambience").alias("ambience"),
    json_str_clean_up_udf("GoodForMeal").alias("goodForMeal"),
    json_str_clean_up_udf("BestNights").alias("bestNights"),
    json_str_clean_up_udf("Music").alias("music")
)

df_parking = spark.read \
    .json(df_json_cleaned_up.rdd.map(lambda r: r.parking)) \
    .drop('_corrupt_record')

df_ambience = spark.read \
    .json(df_json_cleaned_up.rdd.map(lambda r: r.ambience)) \
    .drop('_corrupt_record')

df_goodForMeal = spark.read \
    .json(df_json_cleaned_up.rdd.map(lambda r: r.goodForMeal)) \
    .drop('_corrupt_record')

df_bestNights = spark.read \
    .json(df_json_cleaned_up.rdd.map(lambda r: r.bestNights)) \
    .drop('_corrupt_record')

df_music = spark.read \
    .json(df_json_cleaned_up.rdd.map(lambda r: r.music)) \
    .drop('_corrupt_record')

columns_to_drop = ['attributes', 'hours', 'BusinessParking', 'Ambience', 'GoodForMeal', 'BestNights', 'Music']
df = df.drop(*columns_to_drop)

df_pandas = df.select("*").toPandas()
df_parking_pandas = df_parking.select("*").toPandas()
df_ambience_pandas = df_ambience.select("*").toPandas()
df_goodForMeal_pandas = df_goodForMeal.select("*").toPandas()
df_bestNights_pandas = df_bestNights.select("*").toPandas()
df_music_pandas = df_music.select("*").toPandas()

pd_final_df = pd.concat(
    [df_pandas, df_parking_pandas, df_ambience_pandas, df_goodForMeal_pandas, df_bestNights_pandas, df_music_pandas],
    axis=1
)

# If you want to export the data frame to local csv, uncomment this
# pd_final_df.to_csv('yelp_preprocessed.csv')

db_conn = Engine.get_db_conn()
pd_final_df.to_sql('yelp_raw', con=db_conn, index=False)
