The Yelp dataset that we got is in the json format. Many fields in this dataset cannot 
be parsed easily with any json reader, as they are not json compatible. So many pre-processing
steps were needed to perform in order to read it as pandas data frame, so that we can 
do feature engineering on it.

## Prerequisites
* You need to have a spark cluster and PySpark installed on it.
* MySql RDS connection

## Installation of PySpark
* Apache spark still uses Java 8. So make sure, you have Java 8 before installing PySpark.
* Commands to install PySpark using Anaconda.

`brew install apache-spark`

`conda install -c conda-forge findspark`

`conda install -c conda-forge pyspark`

After pre-processing the json file using PySpark, We saved the final readable .csv file into 
Amazon RDS (MySql) Db.\
This Database will become our source for next steps.

Note: Make sure, you have install mysqlclient package using
`pip install mysqlclient`
    




 