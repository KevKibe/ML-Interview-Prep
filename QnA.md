# ML Questions and Answers

<b>1. How do you handle missing or corrupted data in a data set? </b>
<li>The choice of method depends on factors such as distribution of data, assumptions about the data etc. 
<li>One method is to delete missing values which can result to loss of information depending on the size of deleted data.
<li>Estimating missing values using statistical techniques such as mean, mode and regression imputation.
  
<b>2. Explain the difference between deep learning, artificial intelligence (AI), and machine learning</b>
<li>AI is the development of computer systems that can perform tasks that require human intelligence eg Natural language comprehension,recognizing images
<li>ML is focuses on building algorithms that can learn from data and make predictions using the data.
<li>DL is a field that uses neural networks with multiple layers to uncover complex patterns in data.
  
<b>3. Explain Naive Bayes</b>
  <li>It is an algorithm based on the naive bayes theorem used for classification tasks to predict the class of an observation based on its features.<br>
   It works by calculating the probability of each class given the features and selects the one with the highest probability.<br>
    An example of a usecase of this algorithm is classifying emails as spam or not and document classification with labels like 'sports','culture' 
    
<b>4. What are false positives and false negatives? Why are they significant?</b>
<li>These are terms that describe accuracy of a classification algorithm. False positive is when an observation is wrongly 
  classified as true/present(Type 1 error). False negative is when an observation is wrongly classified as false(Type 2 error). <br>
  The significance depends on the problem eg in medical testing a false positive could lead to unnecessary treatment while a false negative could lead to delayed treatment.<br>
  Or in fraud detection a false negative is more harmful than a false positive.
  
  <b>5. Examples of Supervised Machine Learning used in the world of business today?</b>
<li>Supervised machine learning is used in marketing adn sales to predict customer churn ie when a potential customer is about
  to leave and they use this information to target campaigns for improving retention.

<b>6. xplain the difference between deductive and inductive reasoning in machine learning</b>
<li>Deductive reasoning invlolves deriving conclusions/predictions from general principles/rules. For example a rule-based
  system uses deductive reasoning to state that "if a customer has purchased a product X and Y, then recommend product Z"
<li>Inductive reasoning involves having the model learn patterns and generalizes from the data to make predictions and decisions. <br>
  The difference between the two is that deductive reasoning requires pre-existing knowledge while inductive reasoning requires learning from the data itself.
  
<b>7. How do you know when to use classification or regression</b>
  <li>Classification is mostly used in problems that require labelling or categorizing data instances and the target variable is categorical eg classifying a transaction as fraudulent or not 
  <li>Regression is used when the target variable is a countinuous value eg predicting the price of an item

<b>8. Explain how random forest works</b>
<li>An ensemble of decision trees used for classification and predictive modeling. Instead of relying on  a single decision tree, it combines predictions from multiple decision trees to make accurate predictions.
  Numerous decision tree algorithms are individually trained using different samples from the training dataset in a method called "bagging". Once trained, the random forest takes the same data and feeds it into decision tree, produces a prediction and tallies the results. The most common prediction is selected as teh final prediction. 
    
  
  
  
  
  
  
  
  
  
