import torch
import torch.nn as nn
import torch.nn.functional as F


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
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Ensure input has the right shape
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        # First layer: Linear + ReLU activation
        x = F.relu(self.fc1(x))

        # Output layer: Linear + Softmax activation
        x = F.softmax(self.fc2(x), dim=1)

        return x

    def input(self, flattened_image):
        """
        Process a flattened image through the network.

        Args:
            flattened_image: A 1D tensor of shape (784,) with pixel values normalized to [0,1]

        Returns:
            A list of 10 probabilities corresponding to digits 0-9
        """
        if isinstance(flattened_image, torch.Tensor):
            input_tensor = flattened_image
        else:
            # Convert numpy array to tensor if needed
            input_tensor = torch.tensor(flattened_image, dtype=torch.float32)

        if input_tensor.shape[0] != 784:
            raise ValueError(
                f"Expected input of size 784, got {input_tensor.shape[0]}"
            )

        # Forward pass
        with torch.no_grad():
            predictions = self.forward(input_tensor)

        # Return as a Python list
        return predictions.squeeze().tolist()
