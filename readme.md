# Perceptron for OCR

### Overview
This repository holds the entire perceptron project.
The plan is to build a neural network that can identify handwritten digits (the MNIST dataset).
In order to reach the final level of this project it has been divided into three parts.
Each part representing an increase in complexity with their own respective branch.
We start out small with a single neuron built in a basic python class.
We end up with a neural network that can solve the "Hello World" of machine learning!


## Part 2 (branch: part-2)

### Significant changes made and their impact on the accuracy score:
First model: ~98.2%
Increase to 20 epochs: ~98.1% (epochs 15-17 seem to be right before peak)
Dataset splitting and validating: ~97.1%
Data augmentation: ~96.3%


### Optimizing

On this branch I try to optimize the performance of the model.
Just chasing the highest accuracy score would lead to us choosing a model that has been overfitted and memorized the dataset.
We need to pick something slightly before the peak of the curve.
To do this I decided to use tools such as PyTorch Lightning and WandB.

WandB allows me to view metrics in the browser and compare checkpoints.
In combination with PyTorch Lightning I automatically save a checkpoint both locally and to cloud once a training run is completed.


### How do we determine the best model?

By default we are now testing out 20 epochs (this might change later as i keep testing).
We split up the dataset into training and validation (48k and 12k images respectively).
Once an epoch has finished training on the 48k images, we validate it against the remaining 12k images.
During the entire training process we keep track of which epoch scored the highest on the validation set.
If the next epoch scores higher, we save it and delete the previous one.
We do this 20 times and at the end we are left with the single best performing model.

We then compare these highest scoring models in WandB to pick the best one manually.
We could arguably create a script that picks the one with the highest score but the metrics are right there in front of us in the UI.
It's also a good learning experience to interpret these metrics visually.


### Data augmentation

I implemented augmentation to the dataset.
Types of augmentation being used:
- RandomAffine: Rotation, translation and scaling.
- ElasticTransform: Deformations of the image.
- ColorJitter: Brightness, contrast and saturation.

Notable is that the accuracy score dropped with almost 1% after implementing data augmentation.
This highly suggests that the previous models were slightly overfit.
Previously we tried to anticipate overfitting by keeping track of the accuracy by splitting the dataset and running validation on 10k of the images.
We kept the best performing epoch as out checkpoint based on this metric. It might have been better to atually keep the epoch prior to the highest performing one.
Im going into the next phase with a new mindset and will treat this decrease with 1% as a good thing.






