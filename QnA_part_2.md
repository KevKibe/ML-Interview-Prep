
**How would you define machine learning?**
- a field of computer science that involves designing algorithms and models that enable computers to automatically analyze data
  , recognize patterns, and make predictions or decisions based on the identified patterns. <br>

**What is a labeled training set?**
- a dataset with feature values and the label y
  
**What are the two most common supervised tasks?**
- Supervised(label y is present) 
- Unsupervised (label y is not present)
  
**Can you name four common unsupervised tasks?**
- clustering, anomaly detection, dimensionality reduction, associaion rule learning
  
**What type of algorithm would you use to allow a robot to walk in
  various unknown terrains?**
- Reinforcement Learning
  
**What type of algorithm would you use to segment your customers into
multiple groups?**
- K-means Clustering
  
**Would you frame the problem of spam detection as a supervised
learning problem or an unsupervised learning problem?**
- unsupervised learning

**What is an online learning system?**
- training a system incrementally y feeding it data sequentially either individually or in mini-batches.  
  
**What is out-of-core learning?**
- a machine learning approach that handles large datasets that cannot fit entirely into memory specifically designed to overcome the memory limitations of traditional batch learning algorithms.
  
**What type of algorithm relies on a similarity measure to make
  predictions?**
- Instance Based Learning 
  
**What is the difference between a model parameter and a model
hyperparameter?**
- parameters are estimated based onthe training data and their values define the skill of the model on the problem
- hyperparameters are set manually by the practitioner to help estimate model parameters 
  
**What do model-based algorithms search for? What is the most
  common strategy they use to succeed? How do they make predictions?**
- they use a predictive model that best fits the data. They work by adjusting the parameters and hyperparameters to minimizing a loss function that quantifies the difference between the model's predictions and the actual values or labels in the training data. 
  
**Can you name four of the main challenges in machine learning?**
- Bad Quality Data
- Irrelevant Features
- Insufficient quantity of data
- non-representative data
 
**If your model performs great on the training data but generalizes
poorly to new instances, what is happening? Can you name three
possible solutions?**
- Overfitting on training data. Solutions include Simplifying the model, regularization and getting more training data
  
**What is a test set, and why would you want to use it?**
- A dataset that is unseen by the model. It is used to text the effectiveness/performance of the model

**What is the purpose of a validation set?**
- to evaluate a model during the training process and provide an estimate of its performance on unseen data. 
  
**What is the train-dev set, when do you need it, and how do you use it?**
- training data used to evaluate the modeland is used when there is a possibility that the distribution of the training data is different from the distribution of the data that the model will be applied to. 
  
**What can go wrong if you tune hyperparameters using the test set?**
- It can lead to Overfitting

