from pytorch_version.pytorch_neuron import PytorchNeuralNetwork
from data.mnist_loader import choose_number, get_training_data
import torch
import os


def run_pytorch_version():
    """
    Runs the network if checkpoint exists.
    """
    checkpoint_exists()

    network = PytorchNeuralNetwork()
    checkpoint_path = "checkpoints/checkpoint_epoch_10.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    network.load_checkpoint(checkpoint_path, device)

    data = choose_number(pytorch=True)
    vector = data["vector"].to(device)
    output: list[float] = network.predict(vector)
    print_output(output, data["number"])


def checkpoint_exists():
    """
    Trains the model if the checkpoint does not exist.
    """
    if os.path.exists("checkpoints/checkpoint_epoch_10.pth"):
        return True
    else:
        choice = input(
            "No checkpoint found. Would you like to train the network? (y/n)"
        )
        if choice == "y":
            print("Training the network...")
            run_training()
            return True
        else:
            print("Exiting...")
            exit()


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
    regular = "{0}        {1:8.4f}%        {2:.4e}"

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


def run_training():
    """Trains the network on the training data"""
    # Get the training data
    train_loader = get_training_data()

    network = PytorchNeuralNetwork()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 10
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda")
    else:
        print("Training on CPU")
        device = torch.device("cpu")

    network.training_loop(
        train_loader,
        optimizer,
        criterion,
        num_epochs,
        device,
    )
