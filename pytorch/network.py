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
        conv_channels1=64,  # Number of filters in first conv layer (increased from 32)
        conv_channels2=128,  # Number of filters in second conv layer (increased from 64)
        conv_channels3=256,  # Number of filters in third conv layer (new)
        kernel_size=3,  # Filter size for convolutions
        fc_size=128,  # Size of the fully connected layer after convolutions
    ):
        """
        Initialize a convolutional neural network for image classification.

        Args:
            input_size: Size of the input (784 for MNIST flattened images)
            hidden_size: Size of the hidden layer (kept for compatibility)
            output_size: Size of the output layer (10 for digits 0-9)
            learning_rate: Learning rate for the optimizer
            conv_channels1: Number of filters in first convolutional layer
            conv_channels2: Number of filters in second convolutional layer
            conv_channels3: Number of filters in third convolutional layer
            kernel_size: Size of the convolutional kernel
            fc_size: Size of the fully connected layer after convolutions
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Define the convolutional layers
        # First conv layer: 1 input channel (grayscale) -> conv_channels1 output channels
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels1,
            kernel_size=kernel_size,
            padding=1,
        )
        # Batch normalization after first conv layer
        self.bn1 = nn.BatchNorm2d(conv_channels1)
        # MaxPooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second conv layer: conv_channels1 input channels -> conv_channels2 output channels
        self.conv2 = nn.Conv2d(
            in_channels=conv_channels1,
            out_channels=conv_channels2,
            kernel_size=kernel_size,
            padding=1,
        )
        # Batch normalization after second conv layer
        self.bn2 = nn.BatchNorm2d(conv_channels2)

        # Third conv layer: conv_channels2 input channels -> conv_channels3 output channels
        self.conv3 = nn.Conv2d(
            in_channels=conv_channels2,
            out_channels=conv_channels3,
            kernel_size=kernel_size,
            padding=1,
        )
        # Batch normalization after third conv layer
        self.bn3 = nn.BatchNorm2d(conv_channels3)

        # Calculate the size of the flattened features after convolutions and pooling
        # 28x28 -> after first conv+pool -> 14x14 -> after second conv+pool -> 7x7 -> after third conv+pool -> 3x3
        # So the flattened size will be 3*3*conv_channels3
        self.flat_size = 3 * 3 * conv_channels3

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, fc_size)
        self.fc2 = nn.Linear(fc_size, output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # Define metric
        self.accuracy = Accuracy(task="multiclass", num_classes=output_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier (Glorot) initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

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
            # Single flattened image, reshape to [1, 1, 28, 28]
            x = x.view(1, 1, 28, 28)
        elif x.dim() == 2:
            # Batch of flattened images, reshape to [batch_size, 1, 28, 28]
            batch_size = x.size(0)
            x = x.view(batch_size, 1, 28, 28)
        elif x.dim() == 4 and x.size(1) == 1:
            # Already in the correct format [batch_size, 1, 28, 28]
            pass
        else:
            raise ValueError(f"Unexpected input tensor shape: {x.shape}")

        # First convolutional layer: Conv2d -> BatchNorm -> ReLU -> MaxPool
        x = self.conv1(x)  # Apply convolution
        x = self.bn1(x)  # Apply batch normalization
        x = F.relu(x)  # Apply ReLU activation
        x = self.pool(x)  # Apply max pooling

        # Second convolutional layer: Conv2d -> BatchNorm -> ReLU -> MaxPool
        x = self.conv2(x)  # Apply convolution
        x = self.bn2(x)  # Apply batch normalization
        x = F.relu(x)  # Apply ReLU activation
        x = self.pool(x)  # Apply max pooling

        # Third convolutional layer: Conv2d -> BatchNorm -> ReLU -> MaxPool
        x = self.conv3(x)  # Apply convolution
        x = self.bn3(x)  # Apply batch normalization
        x = F.relu(x)  # Apply ReLU activation
        x = self.pool(x)  # Apply max pooling

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # First fully connected layer with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer
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
        """Configures the optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=5e-6,
        )

        # Add learning rate scheduler that reduces LR when validation accuracy plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # Since we're monitoring accuracy (higher is better)
            factor=0.5,  # Multiply LR by this factor when reducing
            patience=3,  # Number of epochs with no improvement after which LR will be reduced
            verbose=True,  # Print message when LR is reduced
            min_lr=1e-6,  # Lower bound on the learning rate
            threshold=0.0001,  # Minimum change to qualify as an improvement
            threshold_mode="rel",  # Interpret threshold as relative change
        )

        # Return both optimizer and scheduler in the format PyTorch Lightning expects
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",  # The metric to monitor for plateau detection
                "interval": "epoch",  # The scheduler updates after each epoch
                "frequency": 1,  # Update the scheduler every epoch
                "strict": False,  # Don't crash if the monitored metric is missing
            },
        }

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
