from single_neuron.single_neuron import Neuron


def run_single_neuron_version():
    print("\033c")
    # Create a neuron with 3 inputs and sigmoid activation
    print("Creating a neuron with 3 inputs and sigmoid activation...\n")
    neuron = Neuron(num_inputs=3, activation="sigmoid")

    # Display the random weights and bias
    print(f"Random weights: {neuron.weights}")
    print(f"Random bias: {neuron.bias}")

    # Test with sample inputs
    sample_input = [0.5, -0.2, 0.1]
    print(f"\nInput: {sample_input}")

    # Test with different activation functions
    print("\nOutputs:")
    for activation in ["sigmoid", "relu", "leaky_relu", "tanh"]:
        neuron.activation = activation
        result = neuron.forward(sample_input)
        print(f"  {activation}: {result}")
