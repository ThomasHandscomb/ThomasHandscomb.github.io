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

The full details of the different classifiers are included in my repo however one call-out was an interesting comparison of precision vs. recall

<img style = "width:75%; height: auto" src = '/images/NLP_Classifiers.png'>
 	
	</div>
</div>