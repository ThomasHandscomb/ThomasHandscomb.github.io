---
layout: default_ds_blog
title: "A Comparison of some Classifiers"
description: ""
Comment: "A comparison of some common classifiers"
date: 2019-07-12
---

<div id="wrapper">
	<div id="blog-page" class="blogcontainer">


Talk about classifiers here

		<pre> 
			<code class="python">
		#####################################
		# Machine Learning
		# Logistic Regression as a Perceptron
		#####################################

		import numpy as np

		# sigmoid function
		def nonlin(x,deriv=False):
			if(deriv==True):
				return x*(1-x)
			return 1/(1+np.exp(-x))

		nonlin(0.5, True)

		# Seed random numbers
		np.random.seed(1)

		# initialize weights randomly with mean 0
		W = 2*np.random.random((1,1)) - 1
		W
								 
		# Input variables
		X = np.array([[0.23],
						[0.4],
						[0.12],
						[0.32]])
		X
		type(X)
		 
		# output dataset
		Y = np.array([[0,0,1,1]]).T
		Y

		# Naive logistic regression
		from sklearn.linear_model import LogisticRegression
		logistic = LogisticRegression()
		logistic.fit(X,Y)
			</code>
		</pre>

	</div>
</div>