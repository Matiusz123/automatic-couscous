import pandas as pd
from pyspark.sql import SparkSession
#%% md
# Data represents house market information.
# Expects the user to predict the market value of the property using a regression model
#%%
my_spark = SparkSession.builder.getOrCreate()
train_path = 'house-prices-advanced-regression-techniques/train.csv'
test_path = 'house-prices-advanced-regression-techniques/test.csv'
train_data = my_spark.read.csv(train_path,header=True,inferSchema=True)
test_data = my_spark.read.csv(test_path,header=True,inferSchema=True)
#%% md
# First I merge the train and test sets in order to do processing of both
# The Results from the train set is saved in the target variable for later model creation
#%%
from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame

target = train_data.select('Id','SalePrice')
train_data = train_data.drop('SalePrice')

data = DataFrame.unionAll(train_data, test_data)
#%% md
# First we need to replace the values of certain columns because the value 'NA' actually means something
# All the data is explained in the data_description.txt file
#%%
from pyspark.sql.functions import regexp_replace

for column in [
    'Alley',
    'BsmtQual',
    'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'FireplaceQu',
    'GarageType',
    'GarageFinish',
    'GarageQual',
    'GarageCond',
    'PoolQC',
    'Fence',
    'MiscFeature'
]:
    data = data.withColumn(column, regexp_replace(column, 'NA', None))
#%%
int_values_missing = ['LotFrontage','MasVnrArea',
                      'BsmtFinSF1','BsmtFinSF2',
                      'BsmtUnfSF', 'TotalBsmtSF',
                      'BsmtFullBath', 'GarageYrBlt',
                      'BsmtHalfBath', 'GarageCars']

#%%
from pyspark.sql.functions import col,isnan,when,count
data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).show()
