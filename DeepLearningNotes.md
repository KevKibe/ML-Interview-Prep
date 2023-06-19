# Notes on Deep Learning 
- Tensorflow is an open-sorce framework by Google Brain developed for ML and DL. Keras is an API that can be used for building and training DL models.

**Deep Learning**
- Uncovering complex patterns in structured and unstructured data using neural networks with multiple layers.

**Neural Networks**
- Machine learning models inspired by the human brain that uses layers of interconnected nodes/neurons.

**Multi-Layer Perceptron**
- A perceptron is an Artificial neural network with a single layer of nodes/neurons used for linear binary classification.
- A multi-layer perceptron is a feed-foward Artificial Neural Network that has multiple layers of neurons, it can learn non-linear relationships between the input and output variables.
- The data is passed through an input layer and one or more hidden layers the weights are adjusted tominimize the loss and then passed through an activation function to produce an output.

![image](https://github.com/KevKibe/ML-Interview-Prep/assets/86055894/350b39bf-28ff-42d2-925a-279152d0d1cc)

**Data Normalization and why we need it**
- This is a data pre-processing step to scale and transform data soit conforms to a common distribution in order to eliminate data redundancy

**Activation Functions in Neural Networks**
- An activation function is a function that is applied to teh output of a node.

**Sigmoid**
- This maps any input value between 0 and 1 an is used in binary classification.

![image](https://github.com/KevKibe/ML-Interview-Prep/assets/86055894/6b25e5ec-9f44-4881-9ca9-30e9d1b2adc8)

**Tanh**
- It is like the sigmoid but has a steeper slopes and it mapsinput value between -1 an 0.

![image](https://github.com/KevKibe/ML-Interview-Prep/assets/86055894/ee7d0b60-fbf1-454d-9cda-80194a72f058)

**ReLU(Rectified Linear Unit)**
- It maps positive inputs to itself(upto infinity) and any negative values to 0. It is commonly used in the hidden layers 

![image](https://github.com/KevKibe/ML-Interview-Prep/assets/86055894/9bee80b7-2c9f-4d57-8d2c-fbe5b59a6734)

**Leaky ReLU**
- Rectifies the problem in ReLU to map negative inputs. 

![image](https://github.com/KevKibe/ML-Interview-Prep/assets/86055894/cec3a822-b3d7-4f10-93e9-8b8bcaee6dfb)

**Softmax**
- Map input values to a probability distribution of multiple classes eg image classification of a cat or dog.

**Gradient Descent**
- This is an iterative optimization algorithm with the goal of finding a local minimum of a loss function/minimiza the loss function. It works by taking repeated steps in the opposite direction of the gradient of the loss function. The size of the steps is determined by the **learning rate** which is a set hyperparameter by the practitioner.

**Loss/Cost Function**
- **Mean Squared Error** used in regression problems where the goal is to predict a continuous variable
- **Cross-entropy** used in classification problems where the goal is to predict a categorical target variable
- **Hinge-loss** used in binary classification problem to seperate two classes, measures distance between predicted and true class labels
- **Kullback-Leibler (KL) divergence** used to measure difference between 2 probability distributions

**Backpropagation**
- it is a method for computing gradients of model parameters in respect to the loss function, allowing updates of parameters and optimization through gradient descent.
- After data is fed to the network and the loss is calculated, from the outer layer gradients are calcuated for each layer, the gradients represent the sensitivity of the loss function to changes in the parameters of each layer.
- The gradient is calculated and the weights and biases are updated using gradient descent. The steps mentioned are then repeated for a number of epochs until the desired level of performance is achieved. Each iteration involves feeding new input data, computing gradients through backpropagation, and updating the parameters.
- Stochastic Gradient Descent (SGD): In common situations, instead of computing gradients on the entire training dataset, stochastic gradient descent computes gradients on randomly sampled mini-batches of data. This speeds up the computation and allows for more frequent updates.

**Conventional Neural Networks(CNN/ConvNet)**
- These are neural networks with 3 main types of layers Convolution Layer, Pooling Layer and Fully-connected(FC) layer. They are mostly used with image, speech and audio data inputs.  









