from single_neuron.single_pipeline import run_single_neuron_version
from numpy_version.numpy_pipeline import run_numpy_version
from pytorch_version.pytorch_pipeline import run_pytorch_version


def main():
    print("=" * 49)
    print(
        "\033[93m"
        "Neural network demonstration. By Felix Söderström"
        "\033[0m"
    )
    print("=" * 49)
    print(
        "\n\033[90m"
        "On this branch I have built three separate implementations of \n"
        "artificial neurons/networks with increasing complexity to ultimately \n"
        "identify handwritten numbers (the MNIST dataset).\033[0m\n"
    )
    print("\nPlease select one of the following versions:\n")

    return take_input()


def take_input():
    choice = "Banana"
    while choice not in ["1", "2", "3"]:
        print(
            "1. Single neuron   (Floats as input, no backprop)\n"
            "2. NumPy           (Image as input, no backprop)\n"
            "3. PyTorch         (Image as input, with backprop)\n"
        )
        choice = input("Your choice: ")
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
    choice = main()
    print("\033c")
    versions[choice]()
