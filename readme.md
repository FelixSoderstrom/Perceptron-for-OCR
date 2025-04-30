# Perceptron for OCR

### Overview
This repository holds the entire perceptron project.
I have as of today been programming for little over a year now (7 months in python) and now is the first time I get to play around with deep learning!
We are taking this step by step and starting off with a single neuron, at the end we will have a (hopefully) very optimized neural network for the MNIST dataset.
Yes, it's the hello world of machine learning and a great place to start.
There is also a leaderboard online that keeps me motivated to min-max every aspect of this project.

The project is divided into 2 parts (branches) and even smaller steps within those parts (commits).


## Part 1 (branch: part-1)

You are currently on branch: part-2 and can not view the contents of part-1.
Here is a brief summary of what happened in part-1:
- I built a single neuron as a python class.
    The neuron was mostly a random number generator but demonstrates how a single neuron in a neural network works.

- I built a network in NumPy.
    This version took in the flattened data from a single image in the dataset and processed the information through 3 layers.
    This version did not include backprop so arguably, this was a 784-dimensional random number generator.

- I built the same network in PyTorch.
    This version also took in the flattened image and passed it along all layers, btu this time we had backprop.
    We were able to get the models to 98.2% accuracy (as indicated by the validation accuracy mentioned below).


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
Just chasing the highest accuracy score measures in training would result in overfitting, no good.
We need to pick something slightly before the peak of the curve.
To do this I decided to use tools such as PyTorch Lightning and WandB.

WandB allows me to view metrics in the browser and compare checkpoints.
In combination with PyTorch Lightning I automatically save a checkpoint both locally and to cloud once a training run is completed.

I will also explore things like data augmentation, conv layers, hyper-parameter tuning, etc.
Hopefully this will crunch some numbers!


### How do we determine the best model?

By default we are now testing out 20 epochs (this has since changed to 40 epochs with the best performing being between 32 and 36).
We split up the dataset into training and validation (50k/10k respectively).
Once an epoch has finished training on the 50k images we run validation (benchmark) the model against the remaining 10k images while keeping track of the validation accuracy (val_acc).
If model n+1 scores higher than model n, we delete model n. At the end we are left with the single best performing checkpoint of that batch.
We then compare these highest scoring models in WandB to pick the best one manually.

It is absolutely crucial for us to do this cross validation.
Initially I trained the model on all 60k images and then ran validation on the training images.
This resulted in a model that was biased towards the training data. Cheating if you like.
We must not touch the test images until the actual final benchmark! Hence the 50/10 split of the training data.
This 10k set could arguably be cut shorter to increase the amount of data the model is trained on. I might try that later to see if that changes the benchmark accuracy on the test images.

Alongside this I also built my own benchmark to test actual accuracy against the test data.
This benchmark feeds all of the 10k unseen test-images through the model and gives us an accuracy/error-rate.
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


### Additional layers

Changes done to the arrchitectural structure:
- Doubled the amount of channels in the two convolutional layers (from 32 and 64 to 62 and 128)
- Added a third conv layer with 256 channels.
- Increased fully connected layer size to 256 (increase by 100%).

This resulted in a score of 99.54% on the benchmark (increase of 0.02%).

With this I think we have our architecture in place and are ready to move onto other parameters.


### Hyper-parameter tuning

I am dividing the hyper-parameter tuning into two steps.
I will implement the changes in one step, then train and benchmark before moving on to the next step.
But we have finally reached the point where we can start experimenting with the values. I didn't want to do this before we had established a well working network architecture.

Step 1:
I have increased the dropout rate from 0.25 to 0.3 and decreased the weight decay from 1e-5 to 5e-6.
I trained two models and they both scored 99.53% on the benchmark.
This tiny loss is likely within normal variance so i will just go ahead and keep the current parameters and move on to the next step.

Step 2:
I implemented a learning rate scheduler that reduces the learning rate when the validation accuracy peaks.
99.65% on benchmark.


