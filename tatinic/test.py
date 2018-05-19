import pandas as pd
import numpy as np
from pandas import Series, DataFrame
data_train = pd.read_csv("./data/train.csv")
data_train.name='nimei'
print data_train.info()
print data_train.head(5)
print data_train.describe()
print data_train.columns
print data_train.ix[300]
test_train = pd.read_csv("./data/test.csv")
print test_train.info()
print test_train.describe()
