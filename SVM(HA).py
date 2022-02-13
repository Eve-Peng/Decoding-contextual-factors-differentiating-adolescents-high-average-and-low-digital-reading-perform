import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

data=pd.read_csv('./high_average.csv')
y=data[['LABEL']]
x=data.drop(labels=['LABEL'],axis=1)
cv = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=1)

for train_index,test_index in cv.split(x,y):
    X_train, X_test, y_train, y_test = x.iloc[train_index],x.iloc[test_index],y.iloc[train_index],y.iloc[test_index]

    clf=svm.SVC(kernel='linear')
    param_grid={'C':[1]}
    clf_grid = GridSearchCV(clf,param_grid,cv=5,verbose=0,n_jobs=-1)

    clf_grid.fit(X_train,y_train)
    y_pred=clf_grid.predict(X_test)

    print('best parameter:\n',clf_grid.best_params_)

    print('accuracy:', metrics.accuracy_score(y_test, y_pred), 'precision:', metrics.precision_score(y_test, y_pred),
          'recall:', metrics.recall_score(y_test, y_pred),'f-score:', metrics.accuracy_score(y_test, y_pred), 'cm:', metrics.confusion_matrix(y_test, y_pred), 'roc_auc_score:', metrics.roc_auc_score(y_test, y_pred))