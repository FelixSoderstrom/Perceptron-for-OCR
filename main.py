from single_neuron.single_pipeline import run_single_neuron_version
from numpy_version.numpy_pipeline import run_numpy_version
from pytorch_version.pytorch_pipeline import run_pytorch_version

grey_ascii_sequence = "\033[90m"


def main():
    print(
        "\033[93m"
        "Perceptron project built by Felix Söderström\n"
        "\033[90m"
        "In this project I have built three separate implementations of \n"
        "artificial neurons with increasing complexity to ultimately \n"
        "identify handwritten numbers (the MNIST dataset).\n"
        "Yes, it would definately be possible to train the single neuron \n"
        "to identify, lets say a 2, it would only be able to tell you if \n"
        "the image is a 2 or not.\n"
        "For this reason, actual machine learning has only been implemented \n"
        "in the PyTorch version.\n\n\033[0m"
        "Please select one of the following versions:\n"
    )

    return take_input()


def take_input():
    choice = "Banana"
    while choice not in ["1", "2", "3"]:
        print(
            "1. Single neuron (Part 1, not learned)\n"
            "2. NumPy (Part 2, not learned)\n"
            "3. PyTorch (Part 3, learned)\n"
        )
        choice = input("Please input one of the numbers above: ")
        if choice not in ["1", "2", "3"]:
            print("\033c")
            print("\033[91mInvalid input. Please try again.\033[0m")

    return choice


if __name__ == "__main__":
    print("\033c")
    versions = {
        "1": run_single_neuron_version,
        "2": run_numpy_version,
        "3": run_pytorch_version,
    }
    versions[main()]()
