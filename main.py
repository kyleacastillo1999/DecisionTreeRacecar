# @author: Kyle Castillo
# @contact: kylea.castillo1999@gmail.com

# Libraries
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn import metrics
from six import StringIO
import pydotplus

os.environ["PATH"] += 'C:\\Program Files\\Graphviz\\bin'

col_names = ['engine_type',
             'weight',
             'brakes_front',
             'brakes_back',
             'bore_mm',
             'stroke_mm',
             'cooling',
             'drive',
             'electronics',
             'fr_track_in',
             'rr_track_in',
             'frame',
             'fuel_system',
             'fuel_type'
             'material',
             'overall_length_in',
             'width_in',
             'height_in',
             'suspension',
             'tire_brand',
             'tire_dimension_in',
             'wheelbase_in',
             'total_score']

# Loading the data
pima = pd.read_csv("data.csv", header=None, names=col_names)
pima.head()

# Splitting the data into features and the target variable
# Target variable being the total_score
feature_cols = ['engine_type',
                'weight',
                'brakes_front',
                'brakes_back',
                'bore_mm',
                'stroke_mm',
                'cooling',
                'drive',
                'electronics',
                'fr_track_in',
                'rr_track_in',
                'frame',
                'fuel_system',
                'fuel_type'
                'material',
                'overall_length_in',
                'width_in',
                'height_in',
                'suspension',
                'tire_brand',
                'tire_dimension_in',
                'wheelbase_in']


X = pima[feature_cols]
y = pima.total_score

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=1) # 60% test data

clf = DecisionTreeClassifier(criterion="entropy",max_depth=3,splitter="best")

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test,y_pred))