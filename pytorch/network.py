"""
More things I should add:
- Input validation for predict()
    Could allow for more than tensors and just convert them into tensors.

- Validation loop
    The class only has the training loop and doesnt evaluate the model on a separate test.
    This would be useful for to check if the model is learnign properly.

-


"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torchmetrics import Accuracy


class PytorchNeuralNetwork(pl.LightningModule):
    def __init__(
        self,
        input_size=784,
        hidden_size=128,
        output_size=10,
        learning_rate=1e-3,
    ):
        """
        Initialize a simple neural network with one hidden layer.

        Args:
            input_size: Size of the input (784 for MNIST flattened images)
            hidden_size: Size of the hidden layer
            output_size: Size of the output layer (10 for digits 0-9)
            learning_rate: Learning rate for the optimizer
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # Define metric
        self.accuracy = Accuracy(task="multiclass", num_classes=output_size)

        # Initialize weights similar to the NumPy version
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor that can be in multiple formats:
            - Shape (784,) for a single flattened image
            - Shape (batch_size, 784) for batched flattened images
            - Shape (batch_size, 1, 28, 28) for batched MNIST images

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Ensure input has the right shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 4 and x.size(1) == 1:
            x = x.view(x.size(0), -1)
        elif x.dim() == 2 and x.size(1) == 784:
            pass
        else:
            raise ValueError(f"Unexpected input tensor shape: {x.shape}")

        # First layer: Linear + ReLU activation
        x = F.relu(self.fc1(x))

        # Output layer: Linear
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        """Defines the training loop logic for a single batch."""
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Log loss and accuracy
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        acc = self.accuracy(outputs, targets)
        self.log(
            "train_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Defines the validation loop logic for a single batch."""
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        acc = self.accuracy(outputs, targets)

        # Log validation loss and accuracy
        # Important: Use on_epoch=True so the average over the epoch is logged
        # prog_bar=True displays it in the progress bar
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss  # You can return whatever you want, often the loss

    def configure_optimizers(self):
        """Configures the optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )
        return optimizer

    def predict(self, vector: torch.Tensor):
        """
        Process a flattened image through the network.

        Args:
            vector: A 1D tensor of shape (784,) or a batch tensor

        Returns:
            A list of 10 probabilities corresponding to digits 0-9 (for single vector input)
            or a tensor of probabilities (for batch input)
        """
        self.eval()

        if not isinstance(vector, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")

        is_single_vector = False
        if vector.dim() == 1 and vector.shape[0] == 784:
            vector = vector.unsqueeze(0)
            is_single_vector = True
        elif vector.dim() == 2 and vector.shape[1] == 784:
            pass
        elif (
            vector.dim() == 4
            and vector.size(1) == 1
            and vector.size(2) == 28
            and vector.size(3) == 28
        ):
            pass
        else:
            raise ValueError(
                f"Expected input of size (784,) or (batch, 784) or (batch, 1, 28, 28), got {vector.shape}"
            )

        with torch.no_grad():
            logits = self.forward(vector)
            predictions = F.softmax(logits, dim=1)

        if is_single_vector:
            return predictions.squeeze().tolist()
        else:
            return predictions

    @staticmethod
    def load_from_checkpoint_custom(checkpoint_path, device="cpu"):
        """
        Loads a model from a PyTorch Lightning checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file (.ckpt).
            device (str): Device to load the model onto ('cpu' or 'cuda').

        Returns:
            PytorchNeuralNetwork: The loaded model instance.
        """
        model = PytorchNeuralNetwork.load_from_checkpoint(
            checkpoint_path, map_location=device
        )
        model.eval()
        print(
            f"Lightning Checkpoint loaded from {checkpoint_path} to {device}"
        )
        return model
