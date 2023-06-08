# Notes on Deep Learning 
Tensorflow is an open-sorce framework by Google Brain developed for ML and DL. Keras is an API that can be used for building and training DL models.<br>

<b>Deep Learning</b><br>
Uncovering complex patterns in structured and unstructured data using neural networks with multiple layers.

<b>Neural Networks</b><br>
Machine learning models inspired by the human brain that uses layers of interconnected nodes/neurons.

<b>Multi-Layer Perceptron</b><br>
A perceptron is an Artificial neural network with a single layer of nodes/neurons used for linear binary classification.<br>
A multi-layer perceptron is a feed-foward Artificial Neural Network that has multiple layers of neurons, it can learn non-linear relationships between the input and output variables.<br>
The data is passed through an input layer and one or more hidden layers the weights are adjusted tominimize the loss and then passed through an activation function to produce an output.

![image](https://github.com/KevKibe/ML-Interview-Prep/assets/86055894/350b39bf-28ff-42d2-925a-279152d0d1cc)

<b>Data Normalization and why we need it</b><br>
This is a data pre-processing step to scale and transform data soit conforms to a common distribution in order to eliminate data redundancy

<b>Activation Functions in Neural Networks</b><br>
An activation function is a function that is applied to teh output of a node.<br>
<b>Sigmoid</b><br>
This maps any input value between 0 and 1 an is used in binary classification.

![image](https://github.com/KevKibe/ML-Interview-Prep/assets/86055894/6b25e5ec-9f44-4881-9ca9-30e9d1b2adc8)

<b>Tanh</b><br>
It is like the sigmoid but has a steeper slopes and it mapsinput value between -1 an 0.

![image](https://github.com/KevKibe/ML-Interview-Prep/assets/86055894/ee7d0b60-fbf1-454d-9cda-80194a72f058)

<b>ReLU(Rectified Linear Unit)</b><br>
It maps positive inputs to itself(upto infinity) and any negative values to 0. It is commonly used in the hidden layers 

![image](https://github.com/KevKibe/ML-Interview-Prep/assets/86055894/9bee80b7-2c9f-4d57-8d2c-fbe5b59a6734)

<b>Leaky ReLU</b><br>
Rectifies the problem in ReLU to map negative inputs. 

![image](https://github.com/KevKibe/ML-Interview-Prep/assets/86055894/cec3a822-b3d7-4f10-93e9-8b8bcaee6dfb)

<b>Softmax</b><br>
Map input values to a probability distribution of multiple classes eg image classification of a cat or dog.

<b>Gradient Descent</b><br>
This is an iterative optimization algorithm with the goal of finding a local minimum of a loss function/minimiza the loss function. It works by taking repeated steps in the opposite direction of the gradient of the loss function. The size of the steps is determined by the <b>learning rate</b> which is a set hyperparameter by the practitioner.

<b>Loss/Cost Function</b><br>
<b>Mean Squared Error</b> used in regression problems where the goal is to predict a continuous variable<br>
<b>Cross-entropy</b> used in classification problems where the goal is to predict a categorical target variable,br>
<b>Hinge-loss</b> used in binary classification problem to seperate two classes, measures distance between predicted and true class labels<br>
<b>Kullback-Leibler (KL) divergenc</b> used to measure difference between 2 probability distributions


