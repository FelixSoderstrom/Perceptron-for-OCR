from numpy_version.numpy_neuron import NumpyNeuralNetwork
import numpy as np
import os


def run_numpy_version():
    print("\033c")
    n = choose_number()
    flattened_image = get_image(n)
    network = NumpyNeuralNetwork()
    output: list[float] = network.input(flattened_image)
    print_output(output, n)


def print_output(output: list[float], n: int):
    print("\033c")
    print("Number:   Probability:    Full output:")
    for i, o in enumerate(output):
        if i == n:
            print(f"\033[92m{i}        {o*100:8.4f}%        {o}\033[0m")
        else:
            print(f"{i}        {o*100:8.4f}%        {o}")


def choose_number():
    while True:
        try:
            n = int(input("Enter a number between 0-9: "))
            if n < 0 or n > 9:
                raise ValueError
            break
        except ValueError:
            print(
                "\033[91mInvalid input. Please enter a valid number.\033[0m"
            )
    return n


def read_mnist_images(filename):
    """Read MNIST images from IDX file format"""
    with open(filename, "rb") as f:
        # Read header
        magic = int.from_bytes(f.read(4), "big")  # Magic number
        num_images = int.from_bytes(f.read(4), "big")  # Number of images
        rows = int.from_bytes(f.read(4), "big")  # Rows per image
        cols = int.from_bytes(f.read(4), "big")  # Columns per image

        # Read image data
        buffer = f.read(num_images * rows * cols)
        images = np.frombuffer(buffer, dtype=np.uint8)
        return images.reshape(num_images, rows * cols)  # Already flattened


def read_mnist_labels(filename):
    """Read MNIST labels from IDX file format"""
    with open(filename, "rb") as f:
        # Read header
        magic = int.from_bytes(f.read(4), "big")  # Magic number
        num_items = int.from_bytes(f.read(4), "big")  # Number of items

        # Read label data
        buffer = f.read(num_items)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels


def get_image(n: int) -> np.ndarray:
    """
    Get random image data from the test images where the label is equal to n

    Args:
        n: The digit (0-9) to fetch an image for

    Returns:
        A flattened and normalized numpy array of shape (784,)
    """
    # Define file paths
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "mnist_dataset",
    )
    images_file = os.path.join(base_dir, "t10k-images-idx3-ubyte")
    labels_file = os.path.join(base_dir, "t10k-labels-idx1-ubyte")

    # Read test images and labels
    test_images = read_mnist_images(images_file)
    test_labels = read_mnist_labels(labels_file)

    # Find all indices where label matches the requested digit
    matching_indices = np.where(test_labels == n)[0]

    if len(matching_indices) == 0:
        raise ValueError(f"No images found with label {n}")

    # Select a random index from the matching indices
    random_idx = np.random.choice(matching_indices)

    # Get the corresponding image and normalize it (divide by 255 to get values in [0,1])
    image = test_images[random_idx].astype(np.float32) / 255.0

    return image
