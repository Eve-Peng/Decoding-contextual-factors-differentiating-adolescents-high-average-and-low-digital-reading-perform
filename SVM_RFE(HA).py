import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import LinearSVC

LABEL = "LABEL"

data = pd.read_csv("./high_average.csv",header = 0)
datasets = {}
datasets['ID'] = list(data.columns)
datasets['ID'].remove(LABEL)
datasets[LABEL] = np.array(data[LABEL])

featureNames = datasets['ID']
feat, label = data[featureNames], datasets[LABEL]

scaler = preprocessing.StandardScaler()
inputVec = scaler.fit_transform(feat)
tmp, feat = inputVec.copy(), featureNames.copy()
rank = []
score = []
while(tmp.shape[1]):
	clf = LinearSVC()
	clf.fit(tmp,label)
	coef = clf.coef_
	print(coef)
	scores = np.sum(coef**2,axis=0)
	_id_ = np.argmin(scores)
	rank.append(feat[_id_])
	score.append(scores)
	feat.pop(_id_)
	tmp = np.delete(tmp,_id_,axis=1)
print(rank)
print(score)
