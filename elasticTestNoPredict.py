# evaluate an elastic net model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import DataFrame, read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from numpy import arange

# row 550 1,2,258.17999,0.81,4.15606361061035,511.7063,0.23,450.818646616541,1055.35940803383,5736.0,1.22,1.1,0.28,4.31,6.28994,0
# row 784 1,2,154.43,0.528,4.15606361061035,9.44,0.12,450.818646616541,1055.35940803383,5103.0,0.76,0.82,-0.16,4.54,7.27896,100

# load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv('EPA_ml_ready_test.csv')
data = dataframe.values
# impute missing data
my_imputer = SimpleImputer()
data_rev = my_imputer.fit_transform(data)

# # new dataframe
# data_rev_dataframe = DataFrame(data_rev, index=None, columns=None)
# # export dataframe
# data_rev_dataframe.to_csv('imputed_frame.csv', index=False, header=False)

# split into training and test data
X, y = data_rev[:, :-1], data_rev[:, -1]
# define model
model = ElasticNet(alpha=1.0, l1_ratio=0.5)
# fit model
model.fit(X, y)

# define new data
inhabitable = [1,2,258.17999,0.81,4.15606361061035,511.7063,0.23,450.818646616541,1055.35940803383,5736.0,1.22,1.1,0.28,4.31,6.28994]
habitable = [1,2,154.43,0.528,4.15606361061035,9.44,0.12,450.818646616541,1055.35940803383,5103.0,0.76,0.82,-0.16,4.54,7.27896]

# make a prediction
yhat = model.predict([inhabitable])
yhatprime = model.predict([habitable])
# summarize prediction
print('Predicted: %.3f' % yhat)
print('Predicted: %.3f' % yhatprime)