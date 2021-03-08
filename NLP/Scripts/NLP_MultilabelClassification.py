# -*- coding: utf-8 -*-
'''
Last updated: 11/10/2020
by Xinwen
Please also update the instruction txt.

'''
#Include packages. Install pandas, sklearn, collections, numpy if needed.
#For the customize functions, you also need nltk and re

import os
import pandas as pd
from functions import text_prepare, bag_of_words_model, bag_of_words_vector,train_classifier,evaluation_scores
from sklearn.preprocessing import MultiLabelBinarizer
import collections
import numpy as np

#Modify the dictionary size here
dict_size = 100

#navigate to test directory
os.chdir('..')
cwd = os.getcwd()
os.chdir(cwd+'/'+'Test')
cwd = os.getcwd()

#Load train data
#train_data = pd.read_excel (r''+cwd+'\MultiTrain.xlsx')
train_data = pd.read_csv (r''+cwd+'/MultiTrain.csv')
x_train_raw = train_data['Text'].tolist()
y_train_raw = train_data['Label'].tolist()
x_train = []
y_train = []

#Load test data
#test_data = pd.read_excel (r''+cwd+'\MultiTest.xlsx')
test_data = pd.read_csv (r''+cwd+'/MultiTest.csv')
x_test_raw = test_data['Text'].tolist()
y_test_raw = test_data['Label'].tolist()
x_test = []
y_test = []

#Text prepare for labels and texts
for i in x_train_raw:
    i = text_prepare(i)
    x_train.append(i)
    
for i in y_train_raw:
    i = text_prepare(i)
    y_train.append(i)
    
for i in x_test_raw:
    i = text_prepare(i)
    x_test.append(i)
    
for i in y_test_raw:
    i = text_prepare(i)
    y_test.append(i)


#Generate dictionary
words_to_index = bag_of_words_model(x_train,dict_size)
print("Dictionary: " + str(words_to_index))

#bag of words for training text
x_train_vector = []

for i in x_train:
    i = bag_of_words_vector(i,words_to_index,dict_size)
    x_train_vector.append(i)
x_train_vector = np.stack( x_train_vector, axis=0 )

#bag of words for test text
x_test_vector = []
for i in x_test:
    i = bag_of_words_vector(i,words_to_index,dict_size)
    x_test_vector.append(i)
x_test_vector = np.stack( x_test_vector, axis=0 )

#Vectorize labels
tags_counts = {}
tags_counts=collections.Counter(y_train)
mlb = MultiLabelBinarizer(classes = sorted(tags_counts.keys()))
y_train_vector = []
for i in y_train:
    y_train_vector.append(mlb.fit_transform([[i]]))
y_train_vector = np.concatenate( y_train_vector, axis=0 )
y_test_vector=[]
for i in y_test:
    y_test_vector.append(mlb.fit_transform([[i]]))
y_test_vector = np.concatenate( y_test_vector, axis=0 )

#train the classifer
classifier_mybag = train_classifier(x_train_vector, y_train_vector)

#Predicted y_test vector
y_val_predicted_labels_mybag = classifier_mybag.predict(x_test_vector)

#predicted y_test scores
y_val_predicted_scores_mybag = classifier_mybag.decision_function(x_test_vector)

#Predicted y_test labels
y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_mybag)

#Ground truth y_test labels
y_val_inversed = mlb.inverse_transform(y_test_vector)

#Accuracy, F1 score and precision for y_test
accuracy, F1_macro, precision_macro = evaluation_scores(y_test_vector,y_val_predicted_labels_mybag)

#Print the result
print('Prediction accuracy: '+str(accuracy))
print('Prediction F1 score: '+str(F1_macro))
print('Prediction precision: '+str(precision_macro))

#Hold the window
input("Press Enter to Exit...")