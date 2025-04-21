import numpy as np


class NumpyNeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Initialize a simple neural network with one hidden layer.

        Args:
            input_size: Size of the input (784 for MNIST flattened images)
            hidden_size: Size of the hidden layer
            output_size: Size of the output layer (10 for digits 0-9)
        """
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(
            1 / input_size
        )
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(
            1 / hidden_size
        )
        self.b2 = np.zeros(output_size)

    def relu(self, x):
        """ReLU activation function: max(0, x)"""
        return np.maximum(0, x)

    def softmax(self, x):
        """Softmax function for output probabilities"""
        shifted_x = x - np.max(x)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x)

    def forward(self, x):
        """Forward pass through the network"""
        z1 = np.dot(self.W1, x) + self.b1
        a1 = self.relu(z1)

        z2 = np.dot(self.W2, a1) + self.b2
        output = self.softmax(z2)

        return output

    def predict(self, flattened_image: np.ndarray) -> list[float]:
        """
        Process a flattened image through the network.

        Args:
            flattened_image: A 1D numpy array of shape (784,) with pixel values normalized to [0,1]

        Returns:
            A list of 10 probabilities corresponding to digits 0-9
        """
        if flattened_image.shape[0] != 784:
            raise ValueError(
                f"Expected input of size 784, got {flattened_image.shape[0]}"
            )

        if flattened_image.min() < 0:
            flattened_image = (flattened_image + 1) / 2
            # This is just a quick fix.
            # I initially used [0-1] as normalization range
            # but has, since writing this method,
            # moved on to [-1, 1] with the PyTorch network.
            # Both networks reuse the same function to fetch data.
            # I chose to stick with [-1, 1] for consistency.
            # This means that I, ideally,  need to change how we
            # expect the data in forward() instead.
            # I will do this later, this should work for now.

        predictions = self.forward(flattened_image)

        return predictions.tolist()
