# ML Questions and Answers

**How do you handle missing or corrupted data in a data set?**
- The choice of method depends on factors such as distribution of data, assumptions about the data etc. 
- One method is to delete missing values which can result to loss of information depending on the size of deleted data.
- Estimating missing values using statistical techniques such as mean, mode and regression imputation.
  
**Explain the difference between deep learning, artificial intelligence (AI), and machine learning**
- AI is the development of computer systems that can perform tasks that require human intelligence eg Natural language comprehension,recognizing images
- ML is focuses on building algorithms that can learn from data and make predictions using the data.
- DL is a field that uses neural networks with multiple layers to uncover complex patterns in data.
  
**Explain Naive Bayes**
- It is an algorithm based on the naive bayes theorem used for classification tasks to predict the class of an observation based on its features.
- It works by calculating the probability of each class given the features and selects the one with the highest probability.
- An example of a usecase of this algorithm is classifying emails as spam or not and document classification with labels like 'sports','culture' 
    
**What are false positives and false negatives? Why are they significant?**
- These are terms that describe accuracy of a classification algorithm. False positive is when an observation is wrongly 
  classified as true/present(Type 1 error). False negative is when an observation is wrongly classified as false(Type 2 error). 
- The significance depends on the problem eg in medical testing a false positive could lead to unnecessary treatment while a false negative could lead to delayed treatment.
- Or in fraud detection a false negative is more harmful than a false positive.
  
**Examples of Supervised Machine Learning used in the world of business today?**
- Supervised machine learning is used in marketing adn sales to predict customer churn ie when a potential customer is about
  to leave and they use this information to target campaigns for improving retention.

**Explain the difference between deductive and inductive reasoning in machine learning**
- Deductive reasoning invlolves deriving conclusions/predictions from general principles/rules. For example a rule-based
  system uses deductive reasoning to state that "if a customer has purchased a product X and Y, then recommend product Z"
- Inductive reasoning involves having the model learn patterns and generalizes from the data to make predictions and decisions. 
- The difference between the two is that deductive reasoning requires pre-existing knowledge while inductive reasoning requires learning from the data itself.
  
**How do you know when to use classification or regression**
- Classification is mostly used in problems that require labelling or categorizing data instances and the target variable is categorical eg classifying a transaction as fraudulent or not 
- Regression is used when the target variable is a countinuous value eg predicting the price of an item

**Explain how random forest works**
- An ensemble of decision trees used for classification and predictive modeling. Instead of relying on  a single decision tree, it combines predictions from multiple decision trees to make accurate predictions.
- Numerous decision tree algorithms are individually trained using different samples from the training dataset in a method called "bagging". Once trained, the random forest takes the same data and feeds it into decision tree, produces a prediction and tallies the results. The most common prediction is selected as teh final prediction. 
    
  
  
  
  
  
  
  
  
  
