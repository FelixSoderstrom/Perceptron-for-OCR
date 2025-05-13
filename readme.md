# Perceptron for OCR

# Overview

### Introduction

In this repository I will be making my first neural network for the MNIST dataset.
I have been programming for little over a year now (7 months in python) and now is the first time I get to play around with deep learning.
We have divided this into small managable chunks beginning with a single neuron and ending with a optimized neural network for handwritten digit recognition.

This is the second part of the project: Convolutional Neural Network.
In this part I aim to integrate MLOps best practices and optimize the model for the highest accuracy.
So far I have managed to get a 99.65% accuracy while playing around with setting hyperparameters manually.
Optuna managed to get 99.63% accuracy when i let it run for 20 trials.
Even higher scores are possible and I have not exhausted all of the possibilities with Optuna just yet.


### How to run

For docker users:
1. Make sure you have the following installed:
- Docker desktop
- NVIDIA GPU (optional but recommended)
- NVIDIA Drivers (optional but recommended)
- NVIDIA Container Toolkit (optional but recommended)


2. Build and run the container
```bash
docker compose up -d
```

3. Access the Jupyter Notebook by opening your browser and navigating to:

```
http://localhost:8888
```

4. You'll need the token to access the notebook. Get it by running:

```bash
docker logs mnist-ocr
```
Look for a URL with a token parameter, like: `http://127.0.0.1:8888/?token=abc123...`


For windows users:

1. Create a virtual environment (make sure you are using python 3.10)
```bash
python -m venv venv && pip install -r requirements.txt
```

2. Start the notebook and run the cells!



### Project structure

Here are the important files you need to know about:
```
notebook.ipynb
├───┬ checkpoints
|   ├── checkpoint_1.ckpt
|   ├── ...
|   └── checkpoint_n.ckpt
|
├───┬ optuna_results
|   └── study_summary.txt
|
├───┬ wandb
|   └── nested folders for each run
|
├───┬ src
|   ├── config
|   |   └── hyperparameters.py
|   |
|   ├── data
|   |   ├── MNIST
|   |   |   ├── raw
|   |   |   |   ├── dataset files
|   |   |   |   └── ...
|   |   ├── augmentation.py
|   |   └── data_loaders.py
|   |
|   ├── evaluation
|   |   └── evaluation.py
|   |
|   ├── models
|   |   └── network.py
|   |
|   ├── optimization
|   |   └── hyperparameter_tuning.py
|   |
|   ├── training
|   |   └── trainer.py
|   |
|   ├── visualization
|   |   └── visualization.py
|   |
|   └── __init__.py
|
└── more files like readme, venv etc..
```

Explanation: 
- Checkpoints directory. The output folder for out trained models.
- Optuna results directory. The output folder for the Optuna study.
- WandB directory. The output folder for wandb runs.
- src directory. The source code for the project.
    - hyperparameters.py holds all hyperparameters
    - augmentation.py handles data augmentation
    - data_loaders.py handles data loading
    - evaluation.py evaluated the loaded model
    - network.py contains the entire network class
    - hyperparameter_tuning.py handles the hyperparameter tuning using optuna
    - trainer.py sets up trainer and handles training
    - visualization.py handles all visualization in the notebook



# Summary of part 1

You are currently on branch: part-2 and can not view the contents of part-1.
Here is a brief summary of what happened in an earlier episode:
- I built a single neuron as a python class.
    The neuron was mostly a random number generator but demonstrates how a single neuron in a neural network works.

- I built a network in NumPy.
    This version took in the flattened data from a single image in the dataset and processed the information through 3 layers.
    This version did not include backprop so arguably, this was a 784-dimensional random number generator.

- I built the same network in PyTorch.
    This version also took in the flattened image and passed it along all layers, btu this time we had backprop.
    We were able to get the models to 98.2% accuracy (as indicated by the validation accuracy mentioned below).



# How to use

The notebook is the main entrypoint for navigating this project.
Simply open the notebook and run the cells in order.
You might want to create a virtual environment beforehand although this is completely optional.

Running all of the cells with their default settings will:
- Install dependencies
- Import libraries, variables and functions
- Load a pre-trained model
- Evaluate the pre-trained model
- Visualize the data
- Visualize the results of the model

The notebook also allows for training new models and running hyperparameter tuning with Optuna.
The options will be markes as False in the beginning of the notebook. Set either of them to True is you wish to try it out yourself!
Loading a pre-trained model is the only step mentioned above that will be replaces with the option you pick.
All of the evaluation and visualization will happen on your newly trained model if you decide to run all cells automatically.



# Significant changes made to the network

Here I will list significant changes that I've made to the network and how it directly affected the accuracy score.

1. Initial Model: ~98.2%
    I passed all 60k images through a network for 10 epochs.
    This initial model only had fully connected layers.

2. Increased Epochs: ~98.1%
    I increased the total epochs to 20 and saved the last one.
    My model clearly needed more training. I later increased this even further.

3. Dataset Splitting and Validation: ~97.1%
    I implemented cross validation by splitting the dataset into training and validation.
    I traned each epoch on 50k images and then validated the model on the remaining 10k.
    After each validation test I deleted the lower performing model.
    This lowered the accuracy by 1% but this is a good practice we need to implement.
    It gives us the ability to compare models and pick the best one isntead of running benchmarks on every single epoch.

4. Data Augmentation: ~96.3%
    I also created a benchmark in streamlit but this has since been scrapped. Equivalent functionality is found within the notebook after training is complete.
    I applied augmentation to the dataset (see more info below).
    Accuracy dropped by 1% which is probably a sign of overfitting on the previous models.
    I expect this number to increase after implementing convolutional layers.

5. Convolutional Layers: ~99.33%
    This enables our network to "see" the image in 2 dimensions instead of 1.
    The network picks up on features such as egdes and shapes instead of reading individual pixel values of the image.
    This yielded ~3& increase in accuracy! The learning process was also slightly faster.
    This is because of how conv layers work compared to FC layers. In a nutshell, conv layers have less learnable parameters compared to FC layers and therefore the compute time per layer is less.

6. New architecture: ~99.65%
    We now have a network that looks like this: conv1, conv2, conv3, fc1, fc2.
    Previously we only had one convolutional layer.
    This gave us higher accuracy and its the network achitecture I am going to keep throughout the remainder of the project.
    
7. Hyperparameter Tuning: ~99.65%
    I had already been experimenting with manual hyperparameter tuning with the help of LLM's.
    This time I tried Optuna and wow, I should have done that from the start.
    However, we didn't really manage to get any better results.
    This dataset is a very known one and many examples of highly optimized MNIST networks have been documented (and used in LLM training).
    Therefore my consulting with Claude and Chat-GPT has indeed given me highly optimized parameters before even trying out optuna.
    Using optuna was a valuable experience and i will definately use it in the future over manual tuning.


# The project diary

Below here are implementations done to code in the order they happened and any relevant information to these changes.
I talk about the actual changes made, difference in accuracy and why things are the way they are.


### Cross Validation

I want to have a model that performs very well.
But blindly chasing the highest accuracy score on the training data would result in overfitting.
We needed a way to validate how the model performs on unseen data aswell as monitor metrics such as loss and accuracy.
This is where PyTorch Lightning and WandB come in.

WandB allows me to view metrics in the browser and compare checkpoints.
PyTorch streamlines the machine learning process and allows for easy checkpointing.

To prevent overfitting I decided to introduce cross validation.
We split up the dataset into training and validation (50k/10k respectively).
Once an epoch has finished training on the 50k images we run validation on that model against the remaining 10k images while keeping track of the validation accuracy (val_acc).
If epoch2 scores higher than epoch1, we delete epoch1. At the end we are left with the single best performing checkpoint of that batch.
For manual hyperparameter tuning I then compared these checkpoints in the WandB dashboard and picked the best one.
Later on I introduced Optuna to automate this process.

But why split the dataset?
Initially I trained the model on all 60k images and then ran validation on the training images.
This resulted in a model that was biased towards the training data. It gave us a fake accuracy score.
And we must not touch the test images until the final benchmark. Therefore the split of the training data.

Benchmarking (in both the notebook and the old streamlit dashboard) is always done using the test data and the test data is exclusively used for the benchmark!


### Data augmentation

I implemented augmentation to the dataset.
Types of augmentation being used:
- RandomAffine: Rotation, translation and scaling.
- ElasticTransform: Deformations of the image.
- ColorJitter: Brightness, contrast and saturation.

When augmentation was introduced, the accuracy score dropped with almost 1%.
This highly suggests that the previous models were slightly overfit.



### Convolutional layers

We initially made our network a feed forward network and are now switching to convolutional.
This means that we can now process the image in 2 dimensions directly instead of first flattening it.
Obviously this should have been where we started for an efficient workflow.
This type of network also becomes alot less computationally heavy.

Just as expected the accuracy increased and training time increased.
The first two models procured scored 99.40% and 99.41% accuracy respectively on my own benchmark.
Previously, accuracy peaked around epochs 15-17 They are now peaking at 18-20.
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
Manual hyper-parameter tuning.
Here are some things that I tried out
- Increased dropout rate from 0.25 to 0.3
- Decreased weight decay from 1e-5 to 5e-6.
- Implemented LR-scheduler that reduces the learning rate when the validation accuracy peaks.
I trained two models and they both scored ~99.6% on the benchmark.
Tiny losses in between models are likely within normal variance.

Step 2:
Automated hyper-parameter tuning with Optuna.
This actually did not change the accuracy by much and in most cases is actually worse.
This is most likely because I have been asking LLM's for help with understanding the tuning process.
When doing this I have personally become biased towards the language models bias!
Well, I would be stupid not to try what the AI suggests in such a popular dataset as MNIST. And it worked in my favour! Who could have guessed?


### Notebook refactoring

I decided to refactor the python code into a Jupyter notebook.
This was primarily to gather all functionality while still keeping the concerns separated.
It also allows for a much clearer demo this friday.
I also added alot of best practices to the flow in this new version.
Some of the things that are new:
- Dataset visualization before and after augmentation.
- Settings gathered in one place.
- Checkpoint listing
- Model evaluation, confidence and accuracy
- Misclassified examples
- Feature maps from conv layers
- Interactive testing

Claude did most of the refactoring but here is the catch; Claude sort of get a little nosebleed when asked to edit a notebook. I don't know if I'm missing something or if cursors toolset is missing something. There is most likely a fix for this in some underground Github repository.
Instead of taking time to research this I wrote a script that treats the notebook like a JSON (because it is) and then write markdown/codeblocks to it.
Some billionare developer once said: "Lazy people tend to get things done the fastest!". This sort of applies here... 
But no matter the toolset of the agent, AI refactoring will inevitably result in bugs.
However, the network has since been restored to its former glory and all cells now run without tracebacks.



I decided to refactor the project into a notebook.
This took way longer than I would like to admit.
I like to automate these kinds of thigns as much as possible but it just so happens that when you give Claude the instruction to write a notebook he sort of gets a little nosebleed and goes on strike.
I eventually wrote a script that injects the cells into the notebook and handed that tool to my IDE agent.
Everything went sort of okay. I lost some VERY important features in the process that took a while to find.

After all was set and done I realised how stupid this idea was and went back to the old format.
But I had now come to the point where my new structure was nothing like the old one before the notebook. So another refactor was in order..

The current implementation is what matters and here is the summary:
- All machine learning is done outside of the notebook.
- The notebook uses functions defined elsewhere.
- Learning, evaluating, visualizing, hyperparameter tuning can all be done from the notebook.
- The notebook is the primary entrypoint for doing all the above things.
- 
