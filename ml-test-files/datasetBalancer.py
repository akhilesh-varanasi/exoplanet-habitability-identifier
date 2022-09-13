from math import gamma
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import random
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# read and impute csv
data_pd = pd.read_csv('EPA_ml_ready_test.csv')
data = data_pd.values
my_imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
data = my_imputer.fit_transform(data)
# scramble the data
random.shuffle(data)
# separate the target column
main = [row[:-1] for row in data]
target = [row[-1] for row in data]

# oversample underrepresented data
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(main, target)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3,random_state=109)


# create SVC model
clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
print(clf.predict([[1,2,258.17999,0.81,4.15606361061035,511.7063,0.23,450.818646616541,1055.35940803383,5736.0,1.22,1.1,0.28,4.31,6.28994]]))
print(clf.predict([[1,2,154.43,0.528,4.15606361061035,9.44,0.12,450.818646616541,1055.35940803383,5103.0,0.76,0.82,-0.16,4.54,7.27896]]))