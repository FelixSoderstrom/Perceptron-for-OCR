# Perceptron for OCR
### Overview
This repository holds the entire perceptron project.
The plan is to build a neural network that can identify handwritten digits (the MNIST dataset).
In order to reach the final level of this project it has been divided into three parts.
Each part representing an increase in complexity with their own respective branch.
We start out small with a single neuron built in a basic python class.
We end up with a neural network that can solve the "Hello World" of machine learning!

## Part 1 (branch: part-1)
### Single Neuron
Class that represents a single neuron.
Upon instantialization, the neuron will automatically set random weights and bias.
The neuron defaults to sigmoid activation but covers relu, leaky relu and tanh aswell.
When using the main/pipeline scripts, all activations will be outputted on the same instance.
While we could indeed do machine learning on a single neuron, it wouldnt be able to tell me what digit it is.
It could only tell me if the inputted image was the number we trained it on.
This is why I wont be applying any form of back propagation to this version.
But I might come back and do that actually, just because it seems like a really dumb thing to do!

### Numpy
This, like the single neuron class, is a basic implementation of a perceptron.
This class however is a network consisting of several layers that represents the flattened image.
This class therefore takes a test image from the dataset as an input for prediction.
Since the network is not learned it will more or less output random numbers.
But this implementation shows that the image passes through the network and comes out as a prediction.

### PyTorch
This implementation takes the previous numpy implementation and converts it into a pytorch model.
We still feed it one image from the test set and the model spits out a prediction.
The difference here is that this version allows for backprop and can actually learn.
To learn the model, run learn.py from root. All variables are configurable through the pytorch_pipeline.py file.
The current config:
- 10 Epochs
- ADAM Optimizer
- 0.001 Learning Rate
- Cross Entropy Loss
- Trains on GPU if available, otherwise CPU



# To-Do:
I just implemented backprop for the pytorch network.
Next step is to actually load this model and use it for predictions.