from numpy_version.numpy_neuron import NumpyNeuralNetwork
from data.image_getter import choose_number
import numpy as np
import os


def run_numpy_version():
    data = choose_number(numpy=True)
    network = NumpyNeuralNetwork()
    output: list[float] = network.input(data["vector"])
    print_output(output, data["number"])


def print_output(output: list[float], n: int):
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    print("\033c")
    print(
        "Here are the results after passing the "
        f"image through the {BLUE}Numpy{RESET} network"
    )
    print("Note that this network has not been trained!")
    print("\nNumber:   Probability:    Full output:")

    highest = max(output)
    regular = "{0}        {1:8.4f}%        {2}"

    for i, o in enumerate(output):
        p = o * 100  # Percentage
        if i == n and o == highest:  # Correct guess
            print(f"{GREEN}{regular.format(i, p, o)} (Your selection){RESET}")
        elif i == n:
            print(f"{regular.format(i, p, o)} (Your selection)")
        elif o == highest:
            print(f"{GREEN}{regular.format(i, p, o)}{RESET}")
        else:
            print(regular.format(i, p, o))
