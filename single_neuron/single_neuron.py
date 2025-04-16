import random
import math


class Neuron:
    def __init__(self, num_inputs, activation="sigmoid"):
        """
        Initialize a single neuron with random weights and bias.

        Args:
            num_inputs (int): Number of input features
            activation (str): Activation function to use ('sigmoid', 'relu', 'leaky_relu', or 'tanh')
        """
        # Initialize random weights for each input (-1 to 1)
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]

        # Initialize random bias (-1 to 1)
        self.bias = random.uniform(-1, 1)

        # Set activation function
        self.activation = activation

    def _sigmoid(self, x):
        """Sigmoid activation function: 1 / (1 + e^-x)"""
        return 1 / (1 + math.exp(-x))

    def _relu(self, x):
        """ReLU activation function: max(0, x)"""
        return max(0, x)

    def _leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU activation function: max(alpha*x, x)"""
        return x if x > 0 else alpha * x

    def _tanh(self, x):
        """Tanh activation function: (e^x - e^-x) / (e^x + e^-x)"""
        return math.tanh(x)

    def activate(self, x):
        """Apply the selected activation function"""
        if self.activation == "sigmoid":
            return self._sigmoid(x)
        elif self.activation == "relu":
            return self._relu(x)
        elif self.activation == "leaky_relu":
            return self._leaky_relu(x)
        elif self.activation == "tanh":
            return self._tanh(x)
        else:
            raise ValueError(
                f"Unsupported activation function: {self.activation}"
            )

    def forward(self, inputs):
        """
        Compute the output of the neuron for given inputs.

        Args:
            inputs (list): List of input values, must match the number of weights

        Returns:
            float: Output of the neuron after activation
        """
        if len(inputs) != len(self.weights):
            raise ValueError(
                f"Expected {len(self.weights)} inputs, got {len(inputs)}"
            )

        # Calculate the weighted sum of inputs
        weighted_sum = 0
        for i in range(len(inputs)):
            weighted_sum += inputs[i] * self.weights[i]

        # Add bias
        weighted_sum += self.bias

        # Apply activation function and return result
        return self.activate(weighted_sum)
