# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:46:48 2017

@author: vc185059
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from matplotlib.colors import ListedColormap

with open('input3.csv') as f:
    trainingsamples = f.readlines()
    
Xs=[]
Labels=[]
for i in trainingsamples[1:]:
    fs=i.split('\n')
    features=fs[0].split(',')
    Xs.append([float(features[0]),float(features[1])])
    Labels.append(int(features[2]))
#print(Avalues,Bvalues,Labels)


X = np.array(Xs)
y = np.array(Labels)
sss = StratifiedShuffleSplit(y, 1, test_size=0.4, random_state=0)     

for train_index, test_index in sss:
   #print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]


# for coloring based on labels 
# http://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels
#colors = ['red','green']
color= ['red' if l == 0 else 'green' for l in y_train.tolist()]
print(len(X_train),len(X_test),len(Xs))
plt.figure()
plt.scatter(X_train[:,0], X_train[:,1], color=color)
#plt.scatter(X_train[:,0], X_train[:,1], color=y_train.tolist(), cmap=matplotlib.colors.ListedColormap(colors))
#plt.scatter(class0Atrainingvalues, class0Btrainingvalues, color='green', alpha=0.5)
plt.show()
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
f=open('output3.csv','w')
#SVM with Linear Kernel
C = [0.1, 0.5, 1, 5, 10, 50, 100]
from sklearn import svm
from sklearn.model_selection import GridSearchCV
parameters = {'C':C}
svr = svm.SVC(kernel='linear')
clf = GridSearchCV(svr, parameters,cv=5)
clf.fit(X_train, y_train)
bestparameters=clf.best_params_
bestscore=clf.best_score_ 
clf.predict(X_test)
testscore=clf.score(X_test,y_test)

print('svm_linear'+','+str(bestscore)+','+str(testscore)+'\n')
f.write('svm_linear'+','+str(bestscore)+','+str(testscore)+'\n')

##plotting decision boundary

# Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
	Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
		   alpha=0.6)

plt.show()
#plt.set_xlim(xx.min(), xx.max())
#plt.set_ylim(yy.min(), yy.max())
#plt.set_xticks(())
#plt.set_yticks(())
#if ds_cnt == 0:
#	plt.set_title(name)
#plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
#		size=15, horizontalalignment='right')
                
#normed_X_train = normalize(X_train, axis=0, norm='l1')
#normed_X_test = normalize(X_test, axis=0, norm='l1')
normed_X=normalize(X, axis=0, norm='l1')
sss = StratifiedShuffleSplit(y, 1, test_size=0.4, random_state=0)     

for train_index, test_index in sss:
   #print("TRAIN:", train_index, "TEST:", test_index)
   normed_X_train, normed_X_test = normed_X[train_index], normed_X[test_index]
   y_train, y_test = y[train_index], y[test_index]
#plt.figure()
#plt.scatter(normed_X_train[:,0], normed_X_train[:,1], color=color)
##plt.scatter(X_train[:,0], X_train[:,1], color=y_train.tolist(), cmap=matplotlib.colors.ListedColormap(colors))
##plt.scatter(class0Atrainingvalues, class0Btrainingvalues, color='green', alpha=0.5)
#plt.show()
onestoadd=np.ones(len(X_train))
onestoaddfortest=np.ones(len(X_test))
X_train_b=np.vstack((onestoadd,X_train.T)).T
X_test_b=np.vstack((onestoaddfortest,X_test.T)).T
#SVM with Polynomial Kernel
C = [0.1, 1, 3]
degree = [4, 5, 6]
gamma = [0.1, 1]
parameters = {'C':C,'degree':degree,'gamma':gamma}
svr = svm.SVC(kernel='poly')
clf = GridSearchCV(svr, parameters,cv=5)
clf.fit(X_train_b, y_train)
bestparameters=clf.best_params_
bestscore=clf.best_score_
clf.predict(X_test_b)
testscore=clf.score(X_test_b,y_test)
print('svm_polynomial'+','+str(bestscore)+','+str(testscore)+'\n')
f.write('svm_polynomial'+','+str(bestscore)+','+str(testscore)+'\n')

# Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
	Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
		   alpha=0.6)

plt.show()

#SVM with RBF Kernel
C = [0.1, 0.5, 1, 5, 10, 50, 100]
gamma = [0.1, 0.5, 1, 3, 6, 10]
parameters = {'C':C,'gamma':gamma}
svr = svm.SVC(kernel='rbf')
clf = GridSearchCV(svr, parameters,cv=5)
clf.fit(X_train, y_train)
bestparameters=clf.best_params_
bestscore=clf.best_score_
clf.predict(X_test)
testscore=clf.score(X_test,y_test)
print('svm_rbf'+','+str(bestscore)+','+str(testscore)+'\n')
f.write('svm_rbf'+','+str(bestscore)+','+str(testscore)+'\n')

# Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
	Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
		   alpha=0.6)

plt.show()

#Logistic Regression
C = [0.1, 0.5, 1, 5, 10, 50, 100]
parameters = {'C':C}
lr = LogisticRegression()
clf = GridSearchCV(lr, parameters,cv=5)
clf.fit(X_train, y_train)
bestparameters=clf.best_params_
bestscore=clf.best_score_
clf.predict(X_test)
testscore=clf.score(X_test,y_test)
print('logistic'+','+str(bestscore)+','+str(testscore)+'\n')
f.write('logistic'+','+str(bestscore)+','+str(testscore)+'\n')

# Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
	Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
		   alpha=0.6)

plt.show()

#k-Nearest Neighbors
n_neighbors = [i for i in range(1,51)]
leaf_size = [i for i in range(5,61,5)]
parameters = {'n_neighbors':n_neighbors,'leaf_size':leaf_size}
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()
clf = GridSearchCV(neigh, parameters,cv=5)
clf.fit(X_train, y_train)
bestparameters=clf.best_params_
bestscore=clf.best_score_
clf.predict(X_test)
testscore=clf.score(X_test,y_test)
print('knn'+','+str(bestscore)+','+str(testscore)+'\n')
f.write('knn'+','+str(bestscore)+','+str(testscore)+'\n')

# Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
	Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
		   alpha=0.6)

plt.show()

#Decision Trees
max_depth = [i for i in range(1,51)]
min_samples_split = [i for i in range(2,11)]
parameters = {'max_depth':max_depth,'min_samples_split':min_samples_split}
Dt = DecisionTreeClassifier()
clf = GridSearchCV(Dt, parameters,cv=5)
clf.fit(X_train, y_train)
bestparameters=clf.best_params_
bestscore=clf.best_score_
clf.predict(X_test)
testscore=clf.score(X_test,y_test)
print('decision_tree'+','+str(bestscore)+','+str(testscore)+'\n')
f.write('decision_tree'+','+str(bestscore)+','+str(testscore)+'\n')

# Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
	Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
		   alpha=0.6)

plt.show()

#Random Forest
max_depth = [i for i in range(1,51)]
min_samples_split = [i for i in range(2,11)]
parameters = {'max_depth':max_depth,'min_samples_split':min_samples_split}
RFC=RandomForestClassifier()
clf = GridSearchCV(RFC, parameters,cv=5)
clf.fit(X_train, y_train)
bestparameters=clf.best_params_
bestscore=clf.best_score_
clf.predict(X_test)
testscore=clf.score(X_test,y_test)
print('random_forest'+','+str(bestscore)+','+str(testscore)+'\n')
f.write('random_forest'+','+str(bestscore)+','+str(testscore)+'\n')

f.close()

# Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
	Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
# and testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
		   alpha=0.6)

plt.show()

#Xt = np.array(X_train)
#yt = np.array(y_train)
#skf = StratifiedKFold(n_splits=5)
#skf.get_n_splits(Xt, yt)
#
##print(skf)  
#
#for train_index, test_index in skf.split(Xt, yt):
#   print("TRAIN:", len(train_index), "CROSS VALIDATION:", len(test_index))
#   Xt_train, Xt_CV = Xt[train_index], Xt[test_index]
#   yt_train, yt_CV = yt[train_index], yt[test_index]
#
