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
    
**Variance and Bias in ML Models**
- Bias refers to approximating a real-world problem with a simplified model. It is the difference between the average prediction of the model and the true value we are trying to predict. 
- A high bias model leads to underfitting(oversimplifying the underlying patterns in data) which leads to low variance and low performance on the train and test sets beacause the model is too simple for the complex dataset.
- Variance is the measure of sensitivity of a model to changes in the training data. 
- A high variance is more complex and has a greater capacity to fit the training data and capture random fluctuations in the data.
- However, this can lead to overfitting where the model becomes too specific on training data and fails on new unseen data thus low bias and high variance.

**Dimensionality Reduction Techniques**
**Principal Component Analysis**
- An unsupervised learning method that turns high dimension data into low dimension space.
- It identifies direction/ principal components along which data varies the most.
- It achieves dimensionality reduction by projecting data onto selected principal components, discarding components with lower variance.

**t-Distributed Stochastic Neighbour Embedding(t-SNE)**
- a non-linear dimensionality reduction technique used to visualize high dimension data into two or three dimensions.
- it emphasizes local relationships between data points by preserving the similarities and disimilarities between them.
- it uses teh probabilistic approach to model the similarity of data points focusing on preserving the relationships in the lower dimension space.

**Decision Tree Classification**
-a supervised learning algorithm that breaks down the dataset into small subsets while developing a decision tree with branches and nodes.
- pruning in decision trees is a technique that reduces complexity of the classifier by wreducing size of decision trees and hence improving predictive accuracy.

**Kernel SVM**
- an algorithm for pattern analysis. Works by using a kernel function to map the original data into high dimensional space where it  is easier to seperate data using linear classifier.

**Covariance**
- measures the joint variability of two variables and can take any value.

**Correlation**
- is a normalized version of covariance that ranges from -1 to 1. A value of 1 indicates perfect positive linear relationship between the two variables while -1 indicates perfect negative linear relationship. 0 indicates no linear relationship.

**Ensemble Learning**
- A combination of results obtained from ultiple machine learning models to increase accuracy of improved decision-making.

**Cross-Validation**
- a resamping technique that uses different parts of a dataset to train and test aN ML algorithm on different iterations.
- K-Fold Cross Validation is the most popular technique that divides the whole dataset into K sets of equal sizes.

**Evaluation Metrics in Classification**
- Accuracy: measures the proportion of correctly classified instances to the total number of instances. 

- Precision: It calculates the ratio of true positive predictions to the total positive predictions.

- Recall (Sensitivity or True Positive Rate): It calculates the ratio of true positive predictions to the total actual positive instances. It indicates the model's ability to correctly identify positive instances and is useful when the cost of false negatives is high.

- F1-Score: It combines precision and recall into a single metric by taking their harmonic mean. It provides a balanced measure of the model's performance.

- Area Under the ROC Curve (AUC-ROC): It represents the trade-off between the true positive rate and the false positive rate across different probability thresholds. A higher AUC-ROC value indicates better discrimination ability of the model.

- Confusion Matrix: It provides a tabular representation of the model's performance, showing the true positives, true negatives, false positives, and false negatives.

**Evaluation Metrics in Regression**
- Mean Absolute Error (MAE): calculates the average absolute difference between the predicted and actual values. It is less sensitive to outliers compared to the mean squared error.

- Mean Squared Error (MSE): calculates the average squared difference between the predicted and actual values. It amplifies the impact of large errors due to the squaring operation.

- Root Mean Squared Error (RMSE): It is the square root of the mean squared error and provides a measure of the average magnitude of the errors in the predicted values.

- R-squared (Coefficient of Determination): It measures the proportion of the variance in the target variable that can be explained by the model. It ranges from 0 to 1, with a higher value indicating a better fit.

**Clustering Evaluation Metrics**
- Adjusted Rand Index (ARI): It measures the similarity between the clustering results and the ground truth labels, adjusting for chance. It ranges from -1 to 1, with a higher value indicating better clustering.

- Silhouette Coefficient: It evaluates the compactness and separation of clusters based on the average distance between instances within a cluster and the average distance to the nearest neighboring cluster. It ranges from -1 to 1, with a higher value indicating better clustering.

**How do e-commerce websites recommend things to buy?**
- The websites use recommendation systems to suggest products to customers. The systems are implemented through:
  1. Collaborative filtering works by assuming users who bought similar products in the past will repeat the pattern in future.
  2. Content Based filtering uses data from cookies to understand likes and dislikes of each user.
  3. Hybrid filtering which uses both filters.

**How do you design an email spam filter?**
- The approaches to this problem are; using Bayesian filters that compare the content of the email toa database of spam emails, content-based filters that examine the content of the email and look for keywords associated with spam, heuristic filters which just checkthe header.
- Steps when using ML techniques.
1. Understand the problem and establishing the goal as classifying incoming emails as spam or not with high accuracy and minimal false positives and false negatives.
2. Collect data of emails with a labelof either spam or not and train a supervised machine learning algorithm on the data
3. Evaluate and perform cross-validation to select the best model
4. Deployment of teh model and monitoring of performance

**What are the different methods to split a tree in a decision tree algorithm?**
- Reduction in Variance: This method is used in regression decision trees. It measures the reduction in variance achieved by splitting the data using a particular predictor variable. The split that results in the greatest reduction in variance is chosen as the best split
- Gini Impurity in Decision Tree: This is the probability of misclassifying a randomly chosen element in the node if it were randomly labeled according to the distribution of labels in that node. The split that minimizes the Gini index/impurity is chosen as the best split
- Chi-squared Test: This is a test that compares the observed frequency distribution of categorical variable and the expected frequency distribution. It is used to measure the dependency between the target variable and the predictor variables at each potential split. The split with the highest chi-squared value is selected.
- Information Gain: Information gain is based on the concept of entropy, which measures the average amount of information required to classify a sample in a node. It calculates the reduction in entropy achieved by a particular split. The split that maximizes the information gain is selected as the best split.

  
  
  
  
  
  
  
