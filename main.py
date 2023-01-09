from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

Iris = load_iris()
Iris_Data = pd.DataFrame(data=np.c_[Iris['data'], Iris['target']], columns=Iris['feature_names'] + ['target'])
Iris_Data['target'] = Iris_Data['target'].map({0: "setosa", 1: "versicolor", 2: "virginica"})

X_Data = Iris_Data.iloc[:, :-1]
Y_Data = Iris_Data.iloc[:, [-1]]

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

models = []
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('ANN', MLPClassifier()))

from sklearn.metrics import accuracy_score

for name, model in models:
    model.fit(X_Data, Y_Data.values.ravel())
    y_pred = model.predict(X_Data)
    print(name,"'s Accuracy is: ", accuracy_score(Y_Data, y_pred))

from matplotlib import pyplot as plt
from sklearn import model_selection

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_Data, Y_Data.values.ravel(), cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
fig = plt.figure()

# fig.subtitle('Classifier Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()