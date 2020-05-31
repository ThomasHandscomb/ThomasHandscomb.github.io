---
layout: default_ds_blog
title: "NLP Part 2: Classification with Random Forest, Gradient Boosted and Neural Network Models"
description: ""
Comment: "NLP Part 2: Classifying with Random Forest, Gradient Boosted and Neural Network Models"
date: 2019-11-11
---

<div id="wrapper">
	<div id="blog-page" class="blogcontainer">
<h3>Introduction</h3> <br>

<a href = "/2019/11/12/NLP-with-BERT.html"> Part 1 </a> of this NLP post series described a recent project where I was building a binary classification model to try to 
identify competitor products that were similar to our own. <br><br>

The approach I had taken was to reduce the problem to using a single natural language feature of my data to try to describe the labels. <br><br>

<table align="center">
<caption><b>Table 1:</b> Illustrative data</caption>

  <tr>    
    <th>Feature</th>
    <th>Target</th>
  </tr>
  <tr>    
    <td>Word1 Word2 Word3</td>
    <td>1</td>
  </tr>
  <tr>    
    <td>Word4 Word5 Word6 Word7</td>
    <td>0</td>
  </tr>
  <tr>
    <td>Word8 Word9 </td>
    <td>1</td>
  </tr>
  <tr>
    <td>Word11</td>
    <td>0</td>
  </tr>
  <tr>
    <td>...</td>
    <td>...</td>
  </tr>
</table>

<br>

The 2-step process from then was to use Google's BERT model to extract information from this feature to then use as an input into a binary classification model. <br><br>

<img style = "width:80%; height: auto" src = '/images/Transfer_Learning.PNG'>
<br>

This second step is the topic of this post. 

<br><hr><br>
<h3>Implementing in Python</h3> <br>
Link to my GitHub repo: <a href = "https://github.com/ThomasHandscomb/NLP-with-BERT"> NLP-with-BERT </a>
<h4><u>Step 2: Use extracted information to classify </u></h4>
<b> Import modules. </b>

<pre>
	<code class="python">
	#############################################################
	# Title: Binary Classification with BERT and some Classifiers
	# Author: Thomas Handscomb
	#############################################################
	
	# Import libraries
	import matplotlib.pyplot as plt

	import numpy as np
	import pandas as pd

	import torch
	import transformers
	import tensorflow
	import keras

	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import GridSearchCV
	from sklearn.model_selection import cross_val_score
	from sklearn.preprocessing import OneHotEncoder

	from keras.models import Sequential
	from keras.layers import Dense
	</code>
</pre>

The output from the DistilBERT model described in <a href = "/2019/11/12/NLP-with-BERT.html"> Part 1 </a> yields a feature dataframe, features_df that we can use as an input 
against the original target.

<pre>
	<code class="python">
	features_df = pd.DataFrame(np.array(DistilBERT_Output[0][:,0,:]))
	features_df.shape
Out[47]: (692, 768)
	</code>
</pre>

The target label dataframe, labels_df, is constructed from the original data set
<pre>
	<code class="python">
	labels_df = df_Sentiment_Train[['Label']]
	labels_df.shape
Out[48]: (692, 1)
	</code>
</pre>

<b> Create test/train splits </b>
<pre>
	<code class="python">
	train_features_df, test_features_df, train_labels_df, test_labels_df = \
	train_test_split(features_df, labels_df, train_size = 0.75, random_state=40)
	</code>
</pre>

<b> Logistic Regression. </b> The classification report shows precision and recall statistics at the default probability cut-off level (50%). This has an encouraging accuracy level
straight out of the box, however in practise the precision/recall trade-off needs to be tuned to the specific business context. 
<pre>
	<code class="python">
	lr_clf = LogisticRegression()
	lr_clf.fit(train_features_df, train_labels_df)

	from sklearn.metrics import classification_report

	print("Classification On Test Data at default probability threshold (50%)")
	print(classification_report(test_labels_df, lr_clf.predict(test_features_df)))
	</code>
</pre>	

<pre>
	<code class="python">	
	Classification On Test Data at default probability threshold (50%)
              precision    recall  f1-score   support

           0       0.87      0.82      0.84        96
           1       0.79      0.84      0.82        77

    accuracy                           0.83       173
   macro avg       0.83      0.83      0.83       173
weighted avg       0.83      0.83      0.83       173
	</code>
</pre>

<b> Random Forest. </b> We begin by defining a hyper-parameter grid and tuning over this using a brute force search.

<pre>
	<code class="python">
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import RandomizedSearchCV

	# Define random forest classifier model
	rand_forest_model = RandomForestClassifier(n_estimators=1000
							   , max_depth=8
							   , random_state=11
							   , min_samples_leaf=20)

	# Specify hyperparameters to be used in a grid search

	# Num of trees
	n_estimators = [200,400,600,800,1000,1200,1400]
	# Num of features at every split
	max_features = ['auto', 'sqrt']
	# Maximum tree depth
	max_depth = [int(x) for x in np.linspace(5, 25, num = 5)]
	# Minimum number of samples required to split a node
	min_samples_split = [2, 5, 10, 20]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4, 6, 8, 10]
	# Method of selecting samples for training each tree
	bootstrap = [True, False]
	# Out of bag scoring
	oob_score = [True, False]

	hyper_param_grid = {'n_estimators': n_estimators,
				   'max_features': max_features,
				   'max_depth': max_depth,
				   'min_samples_split': min_samples_split,
				   'min_samples_leaf': min_samples_leaf,
				   'bootstrap': bootstrap}
				   #'oob_score': oob_score}

	# Specify a brute force search over hyperparameter grid
	rf_random = RandomizedSearchCV(estimator = rand_forest_model
				   , param_distributions = hyper_param_grid
				   , n_iter = 30
				   , cv = 5
				   , verbose=2
				   , random_state=11
				   , n_jobs = -1)

	# Create hyperparameter training set from the training set
	hyper_train_features_df, hyper_tuning_features_df, \
	hyper_train_labels_df, hyper_tuning_labels_df = \
	train_test_split(train_features_df, train_labels_df, train_size = 0.75, random_state=40)

	# Fit the random forest model on hyperparameter tuning dataset
	rf_random.fit(hyper_tuning_features_df, hyper_tuning_labels_df) 
	</code>
</pre>

Extract best hyper-parameters and define tuned random forest model with these
<pre>
	<code class="python">
	# Best parameters from grid-search
	rf_random.best_params_

	# Define forest model using the tuned hyperparameters
	random_forest_model = \
	RandomForestClassifier(n_estimators = rf_random.best_params_['n_estimators'],
				   min_samples_split=rf_random.best_params_['min_samples_split'],
				   min_samples_leaf = rf_random.best_params_['min_samples_leaf'],
				   max_features = rf_random.best_params_['max_features'],
				   max_depth = rf_random.best_params_['max_depth'],
				   bootstrap = rf_random.best_params_['bootstrap'],
				   #oob_score = rf_random.best_params_['oob_score'],
				   random_state=11,
				   n_jobs=-1)
	# Fit tuned model to the training set
	random_forest_model.fit(train_features_df, train_labels_df)  
	</code>
</pre>

The benefits of the tuning can be seen against a non-tuned model

<pre>
	<code class="python">
	# Compare to a non-tuned version
	basic_random_forest_model = RandomForestClassifier()
	basic_random_forest_model.fit(train_features_df, train_labels_df)

	# Examine simple classifications at default threshold (50%)
	print("Tuned Random Forest On Test Data After CV Grid Search")
	print(classification_report(test_labels_df, random_forest_model.predict(test_features_df)))

	print("Basic Random Forest On Test Data After CV Grid Search")
	print(classification_report(test_labels_df, basic_random_forest_model.predict(test_features_df)))
	</code>
</pre>

<pre>
	<code class="python">
	Tuned Random Forest On Test Data After CV Grid Search
				  precision    recall  f1-score   support

			   0       0.83      0.75      0.79        96
			   1       0.72      0.81      0.76        77

	accuracy                               0.77       173
	macro avg          0.77      0.78      0.77       173
	weighted avg       0.78      0.77      0.78       173

	Basic Random Forest On Test Data After CV Grid Search
				  precision    recall  f1-score   support

			   0       0.75      0.72      0.73        96
			   1       0.67      0.70      0.68        77

	accuracy 	                       0.71       173
	macro avg          0.71      0.71      0.71       173
	weighted avg       0.71      0.71      0.71       173
	</code>
</pre>


<b> Gradient Boosted Model. </b> The Xgboost classifier is straight-forward to instantiate from the xgboost module. We specify the logloss (i.e. binary cross-entropy) metric as
the loss function 
<pre>
	<code class="python">
	from xgboost import XGBClassifier

	# Recall cv/hyper parameter tuning data sets
	#hyper_train_features_df, hyper_tuning_features_df, hyper_train_labels_df, hyper_tuning_labels_df

	xgb_model = XGBClassifier(
		eval_metric='logloss',
		random_seed=11,
		logging_level='Silent',
		nan_mode='Min')
		
	evaluation_set = [(hyper_train_features_df, hyper_train_labels_df), \
	(hyper_tuning_features_df, hyper_tuning_labels_df)]

	xgb_model.fit(hyper_train_features_df
				  , hyper_train_labels_df
				  , eval_set=evaluation_set
				  )

	print("XGboost On Test Data")
	print(classification_report(test_labels_df, xgb_model.predict(test_features_df)))
	</code>
</pre>

<pre>
	<code class="python">
XGboost On Test Data
			  precision    recall  f1-score   support

		   0       0.82      0.68      0.74        96
		   1       0.67      0.82      0.74        77

accuracy      	                       0.74       173
macro avg  	   0.75      0.75      0.74       173
weighted avg       0.75      0.74      0.74       173
	</code>
</pre>

<b> Neural Network. </b> In a sense this is the canonical model to use as a final classification layer as BERT is itself a neural network. Here we essentially append a 
final hidden layer to the BERT output with 768 input neurons and an output layer of 2 neurons for our binary classification, see below. <br>

<img style = "width:50%; height: 75%" src = '/images/nn_model_topology.PNG'>
<br>
We set the activation function to the output layer
as softmax to convert the predictions to probabilities. We choose the Adam optimiser for gradient descent and assign the binary-cross-entropy loss function as with the 
random forest. <br><br>   

I've heavily leveraged the keras package here that provides a nice API to Google's TensorFlow.  
<pre>
	<code class="python">
	nnmodel = Sequential()

	# Add layers - defining number of input and output nodes appropriately
	# Note the softmax activation function on the output layer to convert output to a probability
	nnmodel.add(Dense(2, input_shape=(768,), activation='softmax', name='Output'))

	# Use the Adam optimiser with the binary crossentropy (logloss) metric as the loss function
	nnmodel.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

	# View the network topology
	print('Neural Network Model Summary: ')
	print(nnmodel.summary())

	# Convert labels with one-hot encoding
	encoder = OneHotEncoder(sparse=False)

	train_labels_cat = pd.DataFrame(encoder.fit_transform(train_labels_df).astype(int))
	test_labels_cat = pd.DataFrame(encoder.fit_transform(test_labels_df).astype(int))
	#type(train_labels_cat)
	train_labels_cat.shape
	test_labels_cat.shape

	# Pass DataFrames through the network
	nnmodel_hist = nnmodel.fit(train_features_df, 
				   train_labels_cat, 
				   shuffle=True,
				   #validation_split=0.3, 
				   verbose=2, 
				   batch_size=10, 
				   epochs=20)
	</code>
</pre>

I find the accuracy and loss curves across epochs a thrilling illustration of (albeit fairly simple) machine learning
<pre>
	<code class="python">
	# Summarize history for accuracy
	plt.figure(figsize=(15,7.5))
	plt.plot(nnmodel_hist.history['accuracy'])
	#plt.plot(nnmodel_hist.history['val_acc'])
	plt.title('Model Accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.show()

	# Summarize history for loss
	plt.figure(figsize=(15,7.5))
	plt.plot(nnmodel_hist.history['loss'])
	#plt.plot(nnmodel_hist.history['val_loss'])
	plt.title('Model Loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train'], loc='upper left')
	plt.show()
	</code>
</pre>


<img style = "width:75%; height: auto" src = '/images/nn_model_accuracy.png'>
<img style = "width:75%; height: auto" src = '/images/nn_model_loss.png'>

Of course the network is overfitting on the training data however the simple accuracy across testing classes is still quite good.

<pre>
	<code class="python">
	# Test the network model on testing data
	print("Neural Net Performance on Test Data")
	print(classification_report(test_labels_df, nnmodel.predict_classes(test_features_df)))
	
Neural Net Performance on Test Data
              precision    recall  f1-score   support

           0       0.82      0.93      0.87        96
           1       0.89      0.74      0.81        77

    accuracy                           0.84       173
   macro avg       0.85      0.83      0.84       173
weighted avg       0.85      0.84      0.84       173
	</code>
</pre>

<h4><u>Compare models</u></h4>

The simple classification reports that we have produced so far are useful to some extent however only give a view of the precision/recall tradeoff at the default (50%) probability
level. A more accauracy way of comparing the performance of the different classification models is by constructing and examining the ROC curves and calculating the AUC metrics for
these.
<br><br>
I've included all the code for this in my repository however just show here the final ROC curves for the different classifiers. As expected they all show significant
 uplift over a naive model. 
 <br><br>
It turned out that the logistic regression model performed very well on this
example data set, in practise the random forest and neural net models performed slightly better. Even in this simple example I find it quite thrilling that AUC's of above 90% can 
be achieved purely from the simple string of natural language. <br><br>

<img style = "width:75%; height: auto" src = '/images/NLP_Classifiers.png'>
 	
	</div>
</div>