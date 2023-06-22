# Notes on Deep Learning 
- Tensorflow is an open-sorce framework by Google Brain developed for ML and DL. Keras is an API that can be used for building and training DL models.

## Deep Learning
- Uncovering complex patterns in structured and unstructured data using neural networks with multiple layers.

## Neural Networks
- Machine learning models inspired by the human brain that uses layers of interconnected nodes/neurons.

## Multi-Layer Perceptron
- A perceptron is an Artificial neural network with a single layer of nodes/neurons used for linear binary classification.
- A multi-layer perceptron is a feed-foward Artificial Neural Network that has multiple layers of neurons, it can learn non-linear relationships between the input and output variables.
- The data is passed through an input layer and one or more hidden layers the weights are adjusted tominimize the loss and then passed through an activation function to produce an output.

![image](https://github.com/KevKibe/ML-Interview-Prep/assets/86055894/350b39bf-28ff-42d2-925a-279152d0d1cc)

**Data Normalization and why we need it
- This is a data pre-processing step to scale and transform data soit conforms to a common distribution in order to eliminate data redundancy

## Activation Functions in Neural Networks
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

## Gradient Descent
- This is an iterative optimization algorithm with the goal of finding a local minimum of a loss function/minimiza the loss function. It works by taking repeated steps in the opposite direction of the gradient of the loss function. The size of the steps is determined by the **learning rate** which is a set hyperparameter by the practitioner.

# Loss/Cost Function
- **Mean Squared Error** used in regression problems where the goal is to predict a continuous variable
- **Cross-entropy** used in classification problems where the goal is to predict a categorical target variable
- **Hinge-loss** used in binary classification problem to seperate two classes, measures distance between predicted and true class labels
- **Kullback-Leibler (KL) divergence** used to measure difference between 2 probability distributions

# Backpropagation
- it is a method for computing gradients of model parameters in respect to the loss function, allowing updates of parameters and optimization through gradient descent.
- After data is fed to the network and the loss is calculated, from the outer layer gradients are calcuated for each layer, the gradients represent the sensitivity of the loss function to changes in the parameters of each layer.
- The gradient is calculated and the weights and biases are updated using gradient descent. The steps mentioned are then repeated for a number of epochs until the desired level of performance is achieved. Each iteration involves feeding new input data, computing gradients through backpropagation, and updating the parameters.
- Stochastic Gradient Descent (SGD): In common situations, instead of computing gradients on the entire training dataset, stochastic gradient descent computes gradients on randomly sampled mini-batches of data. This speeds up the computation and allows for more frequent updates.

## Convolutional Neural Networks(CNN/ConvNet)
- These are neural networks with 3 main types of layers Convolution Layer, Pooling Layer and Fully-connected(FC) layer. They are mostly used with image, speech and audio data inputs.
- **Pooling layer:** A layer in a CNN that reduces the spatial dimensions of the data by downsampling it. This helps to reduce the number of parameters in the model and makes it more computationally efficient.

- **Fully connected layer:** A layer in a neural network where each neuron is connected to every neuron in the previous layer. Fully connected layers are often used at the end of a CNN to perform classification or regression on the extracted features.

- **Activation function:** A function applied to the output of a neuron that introduces non-linearity into the model. Common activation functions used in CNNs include the ReLU, sigmoid, and tanh functions.
-  ConvNets work by passing data through a convolution layer which sets learnable filters to the input data to recognize patterns in the data, the output is then passedthrough an activation function to introduce non-linearity. The next layer is the pooling layer which reduces spatial dimensions of the data by downsampling hence reducing parameters and making it more computationally efficient.
- After several rounds of convolution and pooling, the data is passed through one or more fully connected layers to perform prediction. During training, the network learns to optimize its weights using backpropagation and gradient descent.
- An example using the MNIST dataset

```from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.datasets import mnist
from keras.utils import to_categorical

(X_train,y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#one-hot encoding
y_train = to_categorical(y_train)
y_train = to_categorical(y_test)

#creating the model
model = Sequential()
model.add(Conv2D(64, kernel_size=3, activation= 'relu', input_shape(28,28,1)))            #convolution layer
model.add(MaxPooling2D(pool_size=(2,2)))                                                  #max pooling layer
model.add(Conv2D(32, kernel_size=3, activation='relu'))                                   #convolution layer
model.add(Flatten())                                                                      #flattening output of conv layers
moddel.add(Dense(10, activation='relu'))                                                  #fully connected layer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X-train, y_train, validation_data=(X_test, y_test), epochs=3)
```
**Types of CNN Architectures**
1. LeNet - recognizing handwritten and machine-printed characters.
2. GoogLeNet - large scale image recognition.
3. ResNet - large scale image recognition.
5. AlexNet - large scale image recognition.
6. MobileNet - Mobile and embedded vision applications, real-time object detection
7. VGGNet - large scale image recognition.
More [info](https://vitalflux.com/different-types-of-cnn-architectures-explained-examples/)

## Recurrent Neural Networks
- A type of neural network designed to process sequential data and have recurrent connections to allow them to maintain internal state over time. They are used in NLP tasks such as language modelling,text generation.
- It works by processing sequential data one element at a time. At each step,the RNN takes in an input and updates its internal state based on this input and its previous state. The output of the RNN at each time step is calculated based on its updated internal state.
- During training, the RNN learns to make better predictions by adjusting its weights using an algorithm called backpropagation through time.
- Example of RNN code implementation
```
import numpy as np
input_data = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [1.0, 1.1, 1.2],
    [1.3, 1.4, 1.5]
])
target_data = np.array([
    [0.4],
    [0.7],
    [1.0],
    [1.3],
    [1.6]
])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Defining the model
model= Sequential([
    SimpleRNN(64, input_shape=(3, 1)),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(input_data.reshape(5, 3, 1), target_data, epochs=10, batch_size=1)

# Generate predictions
predictions = model.predict(input_data.reshape(5, 3, 1))
print("Predictions:")
print(predictions)
```
**Types of RNN architecture**
1. Long Short-term Memory (LSTM) Networks - addresses the vanishing gradient problem in standard RNNs by using a more complex memory cell structure by effectively capture long-term dependencies in sequential data.
2. Gated Recurrent Unit(GRU) Networks - they capture long-term dependencies. They are similar to LSTMs but have a simplified architecture with fewer gates, making them computationally more efficient.
3. Bidirectional RNN - designed to process inputsequences in forward and backward directions. This captures past and future context which is useful for speech recognition an natural language processing.
4. Encode-Decoder - they consist of 2 RNNS, an encoder that processes the input sequence and produces a fixed-length vector representation and a decoder network that generates the output sequence based on the encoder's representation.
5. Attention Mechanism - a tehnique used to improve RNNS by allowing the network to attend to different parts of the input sequence selectively rather than treating all parts equally.

## Transfer Learning
- A technique where a pre-trained model is used as a startig point for training a new model on a different but related task. For example, a pretrained CNN model trained on large image classification dataset such as Image Net can be used as a feature extractorfor an image classification task with a smaller dataset.
```
from keras.applications.vgg16 import VGG16
# load the pretrained model
model = VGG16(weights='imagenet', include_top = False)
# extract features
features = model.predict_generator(train_generator)

from keras.modes import Sequential
from keras.layers import Dense, Flatten
# flatten the features
features = features.reshape(features.shape[0], -1)
# create a new model
model = Sequential()
# add a Fully connected layer
model.add(dense(1, activation='sigmoid', input_dim=features.shape[1]))
# compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.train(features, train_generator.classes)
```

















 








