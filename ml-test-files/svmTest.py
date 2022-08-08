from sklearn import svm
from random import shuffle
from sklearn.impute import SimpleImputer
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
# from imblearn.over_sampling import SMOTE
# take imputed dataset
# subtract 1 +ve val and 5 -ve val

dataframe = read_csv('EPA_ml_ready_test.csv')
data = dataframe.values
# impute missing data
my_imputer = SimpleImputer()
data_rev = my_imputer.fit_transform(data)
# scramble the data
shuffle(data_rev)
# separate the target column
main = [row[:-1] for row in data_rev]
target = [row[-1] for row in data_rev]
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(main, target, test_size=0.3,random_state=109)
#Create a svm Classifier
clf = svm.SVC(kernel='linear', gamma=2) # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
# predict values
print(clf.predict([[1,2,258.17999,0.81,4.15606361061035,511.7063,0.23,450.818646616541,1055.35940803383,5736.0,1.22,1.1,0.28,4.31,6.28994]]))
print(clf.predict([[1,2,154.43,0.528,4.15606361061035,9.44,0.12,450.818646616541,1055.35940803383,5103.0,0.76,0.82,-0.16,4.54,7.27896]]))