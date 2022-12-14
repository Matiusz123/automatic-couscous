from pyspark.sql import SparkSession, DataFrame, SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from time import time
from pyspark import SparkContext


# Data represents house market information.
# Expects the user to predict the market value of the property using a regression model

my_spark = SparkSession.builder.getOrCreate()
train_path = 'house-prices-advanced-regression-techniques/train.csv'
test_path = 'house-prices-advanced-regression-techniques/test.csv'
train_data = my_spark.read.csv(train_path, header=True, inferSchema=True)
test_data = my_spark.read.csv(test_path, header=True, inferSchema=True)

# First I merge the train and test sets in order to do processing of both
# The Results from the train set is saved in the target variable for later model creation

target = train_data.select('Id', 'SalePrice')
train_data = train_data.drop('SalePrice')

data = DataFrame.unionAll(train_data, test_data)

# First we need to replace the values of certain columns because the value 'NA' actually means something, so we
# replace it with '0' All the data is explained in the data_description.txt file

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
    data = data.withColumn(column, regexp_replace(column, 'NA', '0'))

# Next step is to replace all the missing values that don't mean anything
# with an average value from each column,
# also I'm not sure why the columns changed it;'s type, so let's change them back to desired type


for column in ['LotFrontage', 'MasVnrArea',
               'BsmtFinSF1', 'BsmtFinSF2',
               'BsmtUnfSF', 'TotalBsmtSF',
               'BsmtFullBath', 'GarageYrBlt',
               'BsmtHalfBath', 'GarageCars']:
    me = data.select(mean(col(column)).alias('avg')).collect()[0]['avg']
    me = str(int(me))
    data = data.withColumn(column, regexp_replace(column, 'NA', me))
    data = data.withColumn(column, col(column).cast(IntegerType()))

# Filling all the missing values with the mean values of each column

for column, types in data.dtypes:
    if types == 'int':
        me = data.select(mean(col(column)).alias('avg')).collect()[0]['avg']
        me = int(me)
        data.na.fill(value=me, subset=[column])

# Now we aggregate to find out the count of missing values

data.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in data.columns]).toPandas()

# Now let's perform 3 aggregations with the preprocessed data
# Aggregations :
# - Find average Year of property on the market "YearBuilt"
# - Find mean Lot Area "LotArea"
# - Find skewness of Total square feet of basement area "TotalBsmtSF"

t0 = time()

print("Average Year of property on the market: " + str(data.select(avg("YearBuilt")).collect()[0][0]))
print("Mean Lot Area: " + str(data.select(mean("LotArea")).collect()[0][0]))
print("Skewness of Total square feet of basement area: " + str(data.select(skewness("TotalBsmtSF")).collect()[0][0]))

print("Dataframe aggregations performed in " + str(time() - t0) + " seconds")

# Now let's perform 2 different aggregations with the preprocessed data using groupBy
# Aggregations :
# - Find max Lot Area "LotArea" for each type of "MSZoning": Identifies the general zoning classification of the sale.
# - Find sum of area connected to the street "LotFrontage" for each type of "Neighborhood":
# Physical locations within Ames city limits

t0 = time()
data.groupby('MSZoning').max().select('MSZoning', 'max(LotArea)').show()
data.groupby('Neighborhood').sum().select('Neighborhood', 'sum(LotFrontage)').show()

print("SQL dataframe aggregations with groupby performed in " + str(time() - t0) + " seconds")

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

data.createOrReplaceTempView("sql_table")

t0 = time()
sqlContext.sql('''SELECT MAX(LotArea), MSZoning FROM sql_table GROUP BY MSZoning''').collect()
sqlContext.sql('''SELECT SUM(LotFrontage), Neighborhood FROM sql_table GROUP BY Neighborhood''').collect()

print("SQL aggregations with groupby performed in " + str(time() - t0) + " seconds")
