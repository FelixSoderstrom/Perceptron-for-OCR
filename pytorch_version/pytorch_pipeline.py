from pytorch_version.pytorch_neuron import PytorchNeuralNetwork
from data.image_getter import choose_number


def run_pytorch_version():
    network = PytorchNeuralNetwork()
    data = choose_number(pytorch=True)
    output: list[float] = network.input(data["vector"])
    print_output(output, data["number"])


def print_output(output: list[float], n: int):
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    print("\033c")
    print(
        "Here are the results after passing the "
        f"image through the {RED}PyTorch{RESET} network"
    )
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
