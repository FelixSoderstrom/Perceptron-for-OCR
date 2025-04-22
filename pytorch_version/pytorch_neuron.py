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


class PytorchNeuralNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Initialize a simple neural network with one hidden layer.

        Args:
            input_size: Size of the input (784 for MNIST flattened images)
            hidden_size: Size of the hidden layer
            output_size: Size of the output layer (10 for digits 0-9)
        """
        super(PytorchNeuralNetwork, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

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
        elif x.dim() > 2:
            x = x.view(x.size(0), -1)

        # First layer: Linear + ReLU activation
        x = F.relu(self.fc1(x))

        # Output layer: Linear
        x = self.fc2(x)

        return x

    def predict(self, vector: torch.Tensor):
        """
        Process a flattened image through the network.

        Args:
            flattened_image: A 1D tensor of shape (784,) with pixel values normalized to [0,1]

        Returns:
            A list of 10 probabilities corresponding to digits 0-9
        """
        if isinstance(vector, torch.Tensor):
            input_tensor = vector
        else:
            raise ValueError("Invalid input type")

        if input_tensor.shape[0] != 784:
            raise ValueError(
                f"Expected input of size 784, got {input_tensor.shape[0]}"
            )

        # Forward pass
        with torch.no_grad():
            logits = self.forward(input_tensor)
            predictions = F.softmax(logits, dim=1)

        # Return as a Python list
        return predictions.squeeze().tolist()

    def training_loop(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        num_epochs: int,
        device: torch.device,
        save_frequency: int = 1,
    ):
        """
        Trains the network on the training data.
        Saves checkpoint after each epoch.

        Args:
            train_loader: DataLoader for the training data
            optimizer: Optimizer for the training process
            criterion: Loss function for the training process
            num_epochs: Number of epochs to train the network
            device: Device to train the network on
            save_frequency: Frequency of saving checkpoints
        """
        self.to(device)
        for epoch in range(num_epochs):
            running_loss = 0.0
            total_loss = 0.0
            correct_guesses = 0
            total_guesses = 0

            self.train()

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                running_loss += loss.item()
                total_loss += loss_value

                _, predicted = torch.max(outputs.data, 1)
                total_guesses += targets.size(0)
                correct_guesses += predicted.eq(targets).sum().item()

                if (batch_idx + 1) % 100 == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}]\n"
                        f"Step [{batch_idx+1}/{len(train_loader)}]\n"
                        f"Batch Loss: {total_loss/100:.4f}\n"
                        f"Batch Accuracy: {100 * correct_guesses/total_guesses:.2f}%\n"
                    )
                    running_loss = 0.0

            epoch_loss = total_loss / len(train_loader)
            epoch_accuracy = 100 * correct_guesses / total_guesses

            print(
                f"Epoch [{epoch+1}/{num_epochs}] completed\n"
                f"Loss: {total_loss/100:.4f}\n"
                f"Accuracy: {100 * correct_guesses/total_guesses:.2f}%\n"
            )

            if epoch % save_frequency == 0:
                self.save_checkpoint(
                    epoch, optimizer, epoch_loss, epoch_accuracy
                )

        print("Training complete!")

    def save_checkpoint(
        self, epoch, optimizer, loss, accuracy, dir="checkpoints"
    ):
        """
        Saves a checkpoint of the model and optimizer state.
        """
        os.makedirs(dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
        }
        torch.save(
            checkpoint, os.path.join(dir, f"checkpoint_epoch_{epoch+1}.pth")
        )
        print(f"Checkpoint saved for epoch {epoch+1}")

    def load_checkpoint(self, checkpoint_path, device=None):
        """"""
        if device is None:
            device = next(self.parameters()).device

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
