#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:37:28 2021

@author: root
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Loading the dataset
resumeDataSet = pd.read_csv('/home/aniket/Desktop/flask/Resume_Screening/UpdatedResumeDataSet.csv',encoding='utf-8')
resumeDataSet['cleaned_resume'] = ''
resumeDataSet.head()

print ("Displaying the distinct categories of resume -")
print (resumeDataSet['Category'].unique())

print ("Displaying the distinct categories of resume and the number of records belonging to each category -")
print (resumeDataSet['Category'].value_counts())

from matplotlib.gridspec import GridSpec
targetCounts = resumeDataSet['Category'].value_counts()
targetLabels  = resumeDataSet['Category'].unique()
# Make square figures and axes
plt.figure(1, figsize=(25,25))
the_grid = GridSpec(2, 2)


cmap = plt.get_cmap('coolwarm')
colors = [cmap(i) for i in np.linspace(0, 1, 3)]
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()

import re
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

requiredText = resumeDataSet['cleaned_resume'].values

requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(sublinear_tf = True, stop_words = 'english', max_features = 1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
resumeDataSet.head(20)
clf = OneVsRestClassifier(KNeighborsClassifier())

clf.fit(X_train, y_train)
prediction = clf.predict(X_test)


# Testing_RawData.
message = "Data Science Assurance Associate Data Science Assurance Associate - Ernst & Young LLP Skill Details  JAVASCRIPT- Exprience - 24 months  jQuery- Exprience - 24 months Python- Exprience - 24 monthsCompany Details company - Ernst & Young LLP description - Fraud Investigations and Dispute Services   Assurance TECHNOLOGY ASSISTED REVIEW TAR (Technology Assisted Review) assists in accelerating the review process and run analytics and generate reports. * This too has intelligence to build the pipeline of questions as per user requirement and asks the relevant /recommended questions. Tools & Technologies: Python, Natural language processing, NLTK, spacy, topic modelling, Sentiment analysis, Word Embedding, scikit-learn, JavaScript/JQuery, SqlServer"
Test_df = pd.DataFrame([x.split(';') for x in message.split("\n")])
print(Test_df)
Test_df = Test_df[0].apply(str)
Test_df_clean = Test_df.apply(lambda x: cleanResume(x))
Test_data = word_vectorizer.transform(Test_df_clean)

answer = clf.predict(Test_data)
print(answer)
