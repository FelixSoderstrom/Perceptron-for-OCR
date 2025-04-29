# Perceptron for OCR

### Overview
This repository holds the entire perceptron project.
The plan is to build a neural network that can identify handwritten digits (the MNIST dataset).
In order to reach the final level of this project it has been divided into three parts.
Each part representing an increase in complexity with their own respective branch.
We start out small with a single neuron built in a basic python class.
We end up with a neural network that can solve the "Hello World" of machine learning!


## Part 2 (branch: part-2)

Documentation below will be updated as I progress through the parts.


### Significant changes made and their impact on the accuracy score:

1. Initial Model: ~98.2%
    I passed all 60k images through a network for 10 epochs.

2. Increased Epochs: ~98.1%
    I increased the total epochs to 20 and saved the last one.

3. Dataset Splitting and Validation: ~97.1%
    I split the dataset into training and validation sets.
    I traned each epoch on 48k images and then validated the model on the remaining 12k.
    After each validation test I deleted the lower performing model.

4. Data Augmentation: ~96.3%
    We are from here on out measuring accuracy with the benchmark I made (read more about this under "How do we determine the best model?").
    I applied augmentation to the dataset (see more info below).
    Accuracy dropped by 1% which is probably a sign of overfitting on the previous models.
    I expect this number to increase after implementing convolutional layers.

5. Convolutional Layers: ~99.33%
    Yep, quite a big step up.
    Training also took alot less time.

### Optimizing

On this branch I try to optimize the performance of the model.
Just chasing the highest accuracy score would lead to us choosing a model that has been overfitted and memorized the dataset.
We need to pick something slightly before the peak of the curve.
To do this I decided to use tools such as PyTorch Lightning and WandB.

WandB allows me to view metrics in the browser and compare checkpoints.
In combination with PyTorch Lightning I automatically save a checkpoint both locally and to cloud once a training run is completed.

I will also explore things like data augmentation, conv layers, hyper-parameter tuning, etc.
Hopefully this will crunch some numbers!


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

Alongside this I also built my own benchmark to test actual accuracy against the test data.
This benchmark feeds all of the 10k unseen test-images through the model and gives us an accuracy and error-rate represented in precents.
This whole thing is accessible in the streamlit dashboard.


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


### Convolutional layers

We initially made our network a feed forward network and are now switching to convolutional.
This means that we can now process the image in 2 dimensions directly instead of first flattening it.
Obviously this should have been where we started for an efficient workflow.
This type of network also becomes alot less computationally heavy.

Just as expected the accuracy increased and training time increased.
The first two models procured scored 99.40% and 99.41% accuracy respectively on my own benchmark.
My validation mechanism pulled the last epochs of the training runs, suggesting that the accuracy might even go higher than this.
Previously, accuracy peaked around epochs 15-17.
So lets try more epochs and see if we are leavign decimals on the table.


### Increasing the number of epochs

I learned 3 more models over 25, 30 and 40 epochs.
Saving mechanism saved the checkpoints at epochs 25, 29 and 33 respectively.
Epochs 25 and 29 scored 99.40% accuracy on the benchmark while the 33 scored 99.52%!
I'm not going to spend more time chasing decimal points as the network is not yet done. We will have more time for that later.
But its good to know that we should atleast start with 40 epochs next time.


### Next step
Should be to experiment with different types and numbers of layers aswell as min maxing with hyper-parameter tuning.
