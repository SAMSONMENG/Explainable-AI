import os
import sys
import shutil
import errno
import time
import random
import fnmatch
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn import model_selection
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix
#from xgboost import XGBClassifier
from sklearn.linear_model import Ridge

# combined is 100 lass a and b together
# testadv is not github class a 
# adversarial is bad adversarial that doesn't work
# adv-branch is adversarial ttack with no cache attck trainedo n 9,10 / 17 ,19
#classbsamples class b files for recall

dataset = pd.read_csv('train-consistentsamples.csv') # train chstone/mibench/spec with 100 x86 malware

x = dataset.iloc[:, [0,1,2,3]].values
y = dataset.iloc[:, 4].values

print(x)
print(y)


X_train,X_test, Y_train, Y_test = train_test_split(x,y, test_size= 0.20)

scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  
print(X_train)

print("----first---------------------------------")
#clf = tree.DecisionTreeClassifier(max_depth=10)
#clf = RandomForestClassifier()
#clf = XGBClassifier()
#clf = RandomForestClassifier(max_features=5)
clf = RandomForestClassifier(max_depth=50,n_estimators=50)
#clf = MLPClassifier()
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,50),max_iter=4000, random_state=1)
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
print("-----------------second----------------------")
print("Accuracy: ", np.mean(y_pred==Y_test))


	


#print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
#print(precision_recall_fscore_support(Y_test, y_pred))

# accuracy: (tp + tn) / (p + n)
print("\n")
print("real values \n")


accuracy1 = accuracy_score(y_pred, Y_test)
print('Accuracy: %f' % accuracy1)
# precision tp / (tp + fp)
precision1 = precision_score(y_pred, Y_test)
print('Precision: %f' % precision1)
# recall: tp / (tp + fn)
recall1 = recall_score(y_pred, Y_test)
print('Recall: %f' % recall1)
# f1: 2 tp / (2 tp + fp + fn)
#f1 = f1_score(y_pred, Y_test)
#print('F1 score: %f' % f1)



print('-------------cm------------------')
cm = confusion_matrix(Y_test,y_pred)
#cm
print(cm)

print('------------endcm-------------------')
print("--------------------------first Ridge-----------------------------")
###########################################################



dataset1 = pd.read_csv('spectre1-test-5.csv') #
datasettotallength = 8470
datasetcycles = 5
datasetspilt = 1694

#sample 2 -- readmemorybyte


dataset1 = pd.read_csv('rowhammer-test-1649.csv') #
datasettotallength = 46172
datasetcycles = 1649
datasetspilt = 28

#sample 706 - toggle() - hammers bit

dataset1 = pd.read_csv('zombieload-test-570.csv') #
datasettotallength = 31350
datasetcycles = 570
datasetspilt = 55

#sample 384 - recover() loop


dataset1 = pd.read_csv('flush-test-1231.csv') #
datasettotallength = 49240
datasetcycles = 1231
datasetspilt = 40

#637 - flushandreload() loop

'''
dataset1 = pd.read_csv('meltdown-test-5464.csv') #
datasettotallength = 169384
datasetcycles = 5464
datasetspilt = 31

#3742 - libkdump_read() - read meltdown bit
'''


x1 = dataset1.iloc[:, [0,1,2,3]].values
y1 = dataset1.iloc[:, 4].values
print(x1)

x1 = scaler.transform(x1)
testing = clf.predict(x1)
print("Accuracy---: ", np.mean(testing==y1))
print(x1)

x1 = dataset1.iloc[:, [0,1,2,3]].values
y1 = dataset1.iloc[:, 4].values

#print(x1)
#pertubrations
for i in range(0,len(x1)):
	for j in range(0,4):
		temp1 = x1[i][j]
		temp2 = 0.2 * temp1
		x1[i][j] = temp2 + temp1
	
x2 = x1
'''
rand = random.randint(1,1000)
addition = lambda x: x + rand
print(rand)

x2 = addition(x1)
'''
#-------------------------perterbation----------------
print(x2)


X2 = scaler.transform(x2)
print(X2)
print("-------yes-------x2--------------------------")
y2 = clf.predict(X2)
print("Accuracy: ", np.mean(y1==y2))
print(y2)



clf1 = Ridge()
clf1.fit(X2,y2)

print('first coef: ',clf1.coef_)
x2 = scaler.transform(x2)
print(x2)
coeff = clf1.coef_

for i in range(0, len(x2)):
	for j in range(0,4):
		x2[i][j] = x2[i][j] * coeff[j]

print(x2)

xnew = []
xnewtemp = []
ynew = []
counter = 0
for i in range(0,len(x2)):
	xnewtemp.append(x2[i][0] + x2[i][1] + x2[i][2] + x2[i][3])

#print(xnewtemp)
xnewtemp1 = []
ynewtemp1 = []
for i in range(0,len(xnewtemp)):
	xnewtemp1.append(xnewtemp[i])
	ynewtemp1.append(y2[i])
	if (i+1)%datasetcycles == 0:
		xnew.append(xnewtemp1)
		xnewtemp1 = []
		ynew.append(ynewtemp1)
		ynewtemp1 = []
		
#print(xnew)
#print(len(ynew))

y3 = []
j = 0
k = 0
zero1 = 0
for i in range(datasetspilt):
    for l in range(datasetcycles):
        if ynew[i][l] == 1:
            k = k + 1
        if ynew[i][l] == 0:
            zero1 = zero1 + 1

    if k == zero1:
        print("redo with odd number samples")
        sys.exit()
    if k>zero1:
        y3.append(1)
    else:
        y3.append(0)
    k = 0
    zero1 = 0 

print("--------------------------second Ridge---------------------------")
print(y3)
print('wwe')
Y3 = np.array(y3)
x3 = np.array(xnew)
print("printing before scaling\n")
print(x3)
print(Y3)


print(x3[0])
print(x3[1])
print(x3[2])
print(x3[3])
print(x3[4])
scaler1 = StandardScaler()
scaler1.fit(x3)
X3 = scaler1.transform(x3)
print(X3)
print(X3[0][0])


print("new line ----------------------------\n")
#print(X3[0][3739],X3[0][3740],X3[0][3741],X3[0][3742],X3[0][3743])
#print(X3[1][3739],X3[1][3740],X3[1][3741],X3[1][3742],X3[1][3743])
#print(X3[2][3739],X3[2][3740],X3[2][3741],X3[2][3742],X3[2][3743])
#print(X3[3][3739],X3[3][3740],X3[3][3741],X3[3][3742],X3[3][3743])
#print(X3[4][3739],X3[4][3740],X3[4][3741],X3[4][3742],X3[4][3743])
print("new line ----------------------------\n")

clf2 = Ridge()
clf2.fit(X3,Y3)

print(clf2.coef_)

a = clf2.coef_

b = a[0]
pos = 0
for i in range(0, len(a)):
	if b < a[i]:
		b = a[i]
		pos = i

print(pos + 1)
print(b)
print("coeffcients\n")
print(clf2.coef_[3739])
print(clf2.coef_[3740])
print(clf2.coef_[3741])
print(clf2.coef_[3742])
print(clf2.coef_[3743])
