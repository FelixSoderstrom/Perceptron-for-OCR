1. Introduction and Project Overview
A markdown block explaining the purpose of the notebook, the MNIST dataset, and what we aim to achieve.


2. Imports and Setup
This block imports all required libraries (PyTorch, PyTorch Lightning, Matplotlib, etc.) and sets up basic configurations like random seeds and device selection.


3. Hyperparameter Configuration
A dedicated section for all hyperparameters with clear documentation. This will include model architecture parameters, training parameters, and data parameters.


4. Data Loading Functions
Functions for loading and processing the MNIST dataset, including data augmentation and splitting into training/validation sets.


5. Data Visualization
Code to visualize sample images from the dataset, demonstrating the effects of data augmentation.


6. Neural Network Model Definition
The implementation of the PyTorch Lightning model class with convolutional layers, fully connected layers, and all required methods.


7. Training Configuration and Setup
Setup of WandB logging, model checkpoint callbacks, and training configuration.


8. Training Execution
The actual training code that initializes the network and trains it using PyTorch Lightning.


9. Checkpoint Management Functions
Functions to find, load, and manage model checkpoints.


10. Model Evaluation
Code to evaluate the trained model on the validation/test set, including accuracy metrics.


11. Prediction Visualization
Functions to visualize predictions made by the model on sample digits.


12. Interactive Testing
An interactive section allowing users to test the model on custom inputs or selected test samples.


13. Summary and Results
A markdown section summarizing the results, achieved accuracy, and potential improvements.


14. Utility Functions
Additional helper functions for displaying outputs, finding best checkpoints, etc.