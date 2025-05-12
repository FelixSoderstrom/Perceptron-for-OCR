import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from src.data.data_loader import (
    load_test_dataset,
    get_digit_image,
    get_dataset,
)

from src.config.hyperparameters import hyperparameters as hp


def visualize_predictions(model, num_samples=10):
    """
    Visualize predictions from the model on random test samples.

    Args:
        model: Trained model to use for predictions
        num_samples: Number of samples to visualize

    Returns:
        plt: Matplotlib figure with the visualizations
    """
    # Load test data
    test_loader = load_test_dataset(batch_size=num_samples)

    # Get a batch of test images
    images, labels = next(iter(test_loader))

    # Move to same device as model
    device = next(model.parameters()).device
    images, labels = images.to(device), labels.to(device)

    # Get model predictions
    with torch.inference_mode():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)

    # Convert to numpy for plotting
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    probabilities = probabilities.cpu().numpy()

    # Create figure with subplots
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 1.5, 4))

    # Plot each image and its prediction
    for i in range(num_samples):
        # Display the image
        img = images[i].squeeze()
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"True: {labels[i]}")
        axes[0, i].axis("off")

        # Display the prediction probabilities as a bar chart
        axes[1, i].bar(range(10), probabilities[i])
        axes[1, i].set_xticks(range(10))
        axes[1, i].set_ylim(0, 1)
        if predicted[i] == labels[i]:
            color = "green"
        else:
            color = "red"
        axes[1, i].set_title(f"Pred: {predicted[i]}", color=color)

    plt.tight_layout()
    return plt


def plot_confusion_matrix(model, normalize=False):
    """
    Generate a confusion matrix for the trained model on the test dataset.

    Args:
        model: Trained model to evaluate
        normalize: Whether to normalize the confusion matrix

    Returns:
        plt: Matplotlib figure with the confusion matrix
    """
    # Load test data
    test_loader = load_test_dataset(batch_size=1000)

    # Get device
    device = next(model.parameters()).device

    # Initialize lists to store predictions and ground truth
    all_preds = []
    all_labels = []

    # Get predictions for all test samples
    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Normalize if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create display
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=range(10)
    )
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion matrix")

    return plt


def visualize_feature_maps(model, digit=3):
    """
    Visualize feature maps from each convolutional layer for a given digit.

    Args:
        model: Trained model to extract feature maps from
        digit: Which digit to visualize (0-9)

    Returns:
        plt: Matplotlib figure with the feature maps
    """
    # Get an image of the specified digit
    img_tensor, orig_img = get_digit_image(digit)

    # Move to same device as model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    # Preprocess for model
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Set model to eval mode
    model.eval()

    # Lists to store activations for each layer
    activations = []

    # Register hooks to capture outputs of each conv layer
    handles = []

    def hook_fn(module, input, output):
        activations.append(output.detach())

    # Register the hook for each convolutional layer
    for conv_layer in model.conv_layers:
        for layer in conv_layer:
            handles.append(layer.register_forward_hook(hook_fn))

    # Forward pass to get activations
    with torch.inference_mode():
        _ = model(img_tensor)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Create figure for visualization
    fig = plt.figure(figsize=(15, 10))

    # Plot the original image
    plt.subplot(1, 1, 1)
    plt.imshow(orig_img.squeeze(), cmap="gray")
    plt.title(f"Original Image (Digit {digit})")
    plt.axis("off")

    # Create separate figures for each layer's feature maps
    for i, layer_activation in enumerate(activations):
        # Move to CPU and convert to numpy
        layer_activation = layer_activation.cpu().numpy()[
            0
        ]  # Remove batch dimension

        # Determine number of feature maps to show (up to max)
        max_maps_per_layer = 8  # Adjust as needed
        num_maps = min(max_maps_per_layer, layer_activation.shape[0])

        # Create a new figure for this layer
        plt.figure(figsize=(10, 10))
        plt.suptitle(
            f"Layer {i+1} Feature Maps ({layer_activation.shape[0]} channels)"
        )

        # Create a grid to place feature maps
        grid_size = int(np.ceil(np.sqrt(num_maps)))

        # Plot each feature map
        for j in range(num_maps):
            plt.subplot(grid_size, grid_size, j + 1)

            # Normalize the feature map for better visualization
            feature_map = layer_activation[j]
            feature_map = (feature_map - feature_map.min()) / (
                feature_map.max() - feature_map.min() + 1e-10
            )

            plt.imshow(feature_map, cmap="viridis")
            plt.axis("off")

        plt.tight_layout()

    # Return to the original figure
    plt.figure(fig.number)

    # # Create figure for visualization
    # fig = plt.figure(figsize=(15, 10))

    # # Plot the original image
    # plt.subplot(3, 1, 1)
    # plt.imshow(orig_img.squeeze(), cmap="gray")
    # plt.title(f"Original Image (Digit {digit})")
    # plt.axis("off")

    # # For each layer, plot a subset of feature maps
    # max_maps_per_layer = 8  # Adjust as needed

    # for i, layer_activation in enumerate(activations):
    #     # Move to CPU and convert to numpy
    #     layer_activation = layer_activation.cpu().numpy()[
    #         0
    #     ]  # Remove batch dimension

    #     # Determine number of feature maps to show (up to max)
    #     num_maps = min(max_maps_per_layer, layer_activation.shape[0])

    #     plt.subplot(len(activations) + 1, 1, i + 2)
    #     plt.title(
    #         f"Layer {i+1} Feature Maps ({layer_activation.shape[0]} channels)"
    #     )

    #     # Create a grid to place feature maps
    #     grid_size = int(np.ceil(np.sqrt(num_maps)))

    #     # Plot each feature map
    #     for j in range(num_maps):
    #         plt.subplot(grid_size, grid_size, j + 1)

    #         # Normalize the feature map for better visualization
    #         feature_map = layer_activation[j]
    #         feature_map = (feature_map - feature_map.min()) / (
    #             feature_map.max() - feature_map.min() + 1e-10
    #         )

    #         plt.imshow(feature_map, cmap="viridis")
    #         plt.axis("off")

    plt.tight_layout()
    return plt


def predict_single_digit(model, digit=None):
    """
    Make a prediction on a single digit and visualize the results.

    Args:
        model: Trained model to use for prediction
        digit: Specific digit to predict (0-9), or None for random

    Returns:
        plt: Matplotlib figure with the visualization
    """
    # If no digit specified, choose random digit
    if digit is None:
        digit = np.random.randint(0, 10)

    # Get an image of the specified digit
    img_tensor, orig_img = get_digit_image(digit)

    # Move to same device as model
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    # Get model prediction
    model.eval()
    with torch.inference_mode():
        probabilities = model.predict(img_tensor)
        prediction = torch.tensor(probabilities).argmax().item()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Plot the image
    ax1.imshow(orig_img.squeeze(), cmap="gray")
    ax1.set_title(f"Test Digit (True Label: {digit})")
    ax1.axis("off")

    # Plot the prediction probabilities
    colors = ["gray"] * 10
    colors[prediction] = "green" if prediction == digit else "red"

    ax2.bar(range(10), probabilities, color=colors)
    ax2.set_xticks(range(10))
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Probability")
    ax2.set_title(f"Prediction: {prediction}")

    plt.tight_layout()
    return plt


def visualize_errors(model, error_indices, n_examples=10):
    """
    Visualize examples that the model misclassified.

    Args:
        model: Trained model
        error_indices: List of tuples (index, true_label, predicted_label)
        n_examples: Number of examples to visualize
    """
    # Load the test dataset for visualization
    test_dataset = get_dataset(train=False, transform=transforms.ToTensor())

    # Select a subset of errors to visualize
    n_errors = min(n_examples, len(error_indices))
    sample_errors = error_indices[:n_errors]

    # Prepare the figure
    fig, axes = plt.subplots(2, n_errors // 2 + n_errors % 2, figsize=(15, 6))
    axes = axes.flatten()

    # For each error example
    for i, (idx, true_label, pred_label) in enumerate(sample_errors):
        # Get the image
        img, _ = test_dataset[idx]

        # Display the image
        axes[i].imshow(img.squeeze().numpy(), cmap="gray")
        axes[i].set_title(f"True: {true_label}, Pred: {pred_label}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def plot_digit_probabilities(model, digit_idx, test_dataset=None):
    """
    Plot the probability distribution for a single digit.

    Args:
        model: Trained model to use for prediction
        digit_idx: Index of the digit in the test dataset
        test_dataset: Test dataset, created if None
    """
    # Load test dataset if not provided
    if test_dataset is None:
        test_dataset = get_dataset(
            train=False, transform=transforms.ToTensor()
        )

    # Get the image and label
    img, label = test_dataset[digit_idx]
    img_tensor = img.unsqueeze(0).to(hp["device"])

    # Set the model to evaluation mode
    model.eval()

    # Get predictions
    with torch.inference_mode():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)

    # Convert to numpy for plotting
    probs_np = probs.cpu().numpy().flatten()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the digit
    ax1.imshow(img.squeeze().numpy(), cmap="gray")
    ax1.set_title(f"Digit (True Label: {label})")
    ax1.axis("off")

    # Plot probability distribution
    bar_colors = ["C0"] * 10
    pred_label = np.argmax(probs_np)
    bar_colors[pred_label] = "C3" if pred_label != label else "C2"

    ax2.bar(range(10), probs_np, color=bar_colors)
    ax2.set_xticks(range(10))
    ax2.set_xlabel("Digit Class")
    ax2.set_ylabel("Probability")
    ax2.set_title("Prediction Probabilities")
    ax2.set_ylim([0, 1])

    # Add a horizontal line at 0.1 (random chance would be 0.1)
    ax2.axhline(y=0.1, color="r", linestyle="--", alpha=0.3)

    # Annotate the highest probability
    max_prob = np.max(probs_np)
    ax2.annotate(
        f"{max_prob:.4f}",
        xy=(pred_label, max_prob),
        xytext=(pred_label, max_prob + 0.05),
        ha="center",
    )

    plt.tight_layout()
    plt.show()

    return probs_np


def create_digit_grid(model, test_dataset=None, grid_size=5):
    """
    Create a grid of digits with their predictions.

    Args:
        model: Trained model to use for prediction
        test_dataset: Test dataset, created if None
        grid_size: Size of the grid (grid_size x grid_size)

    Returns:
        Figure object for the grid
    """
    # Load test dataset if not provided
    if test_dataset is None:
        test_dataset = get_dataset(
            train=False, transform=transforms.ToTensor()
        )

    # Create a figure and axes
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    # Set the model to evaluation mode
    model.eval()

    # For each cell in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Get a random sample
            idx = np.random.randint(0, len(test_dataset))
            img, label = test_dataset[idx]

            # Get model prediction
            with torch.inference_mode():
                img_tensor = img.unsqueeze(0).to(hp["device"])
                logits = model(img_tensor)
                probs = F.softmax(logits, dim=1)
                pred_label = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_label].item()

            # Display the image
            axes[i, j].imshow(img.squeeze().numpy(), cmap="gray")

            # Create colored border based on prediction correctness
            if pred_label == label:
                border_color = "green"
            else:
                border_color = "red"

            # Add colored border
            for spine in axes[i, j].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)

            # Add title with prediction
            axes[i, j].set_title(
                f"P: {pred_label} ({confidence:.2f})\nT: {label}"
            )
            axes[i, j].axis("off")

    plt.tight_layout()
    return fig


def print_summary(summary):
    """
    Print a formatted summary of model performance.

    Args:
        summary: Dictionary of summary metrics
    """
    print("=" * 50)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 50)

    print(f"\nOverall Accuracy: {summary['accuracy']:.4f}%")
    print(f"Error Rate: {summary['error_rate']:.4f}%")

    print(f"\nTotal Samples: {summary['n_samples']}")
    print(
        f"Correct Predictions: {summary['n_samples'] - summary['n_errors']}"
    )
    print(f"Incorrect Predictions: {summary['n_errors']}")

    print("\nPer-class Performance:")
    for i, acc in enumerate(summary["class_accuracies"]):
        print(f"  Digit {i}: {acc:.4f}%")

    print(
        f"\nBest Performing Digit: {summary['best_class']} ({summary['class_accuracies'][summary['best_class']]:.4f}%)"
    )
    print(
        f"Worst Performing Digit: {summary['worst_class']} ({summary['class_accuracies'][summary['worst_class']]:.4f}%)"
    )

    print("\nMost Confused Digit Pairs:")
    for true_digit, pred_digit, count in summary["confused_pairs"]:
        print(
            f"  True: {true_digit}, Predicted: {pred_digit}, Count: {count}"
        )

    print("\nModel Architecture:")
    arch = summary["model_architecture"]
    print(f"  Convolutional Layers: {arch['conv_layers']}")
    print(f"  Fully Connected Layers: {arch['fc_layers']}")
    print(f"  Total Parameters: {arch['total_parameters']:,}")

    if isinstance(arch["avg_inference_time"], float):
        print(
            f"  Average Inference Time: {arch['avg_inference_time'] * 1000:.2f} ms per batch"
        )

    print("\nHardware:")
    if torch.cuda.is_available():
        print(f"  Device: GPU ({torch.cuda.get_device_name(0)})")
        print(
            f"  Memory Usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )
    else:
        print("  Device: CPU")

    print("=" * 50)


def plot_summary_graphs(summary):
    """
    Plot graphs summarizing model performance.

    Args:
        summary: Dictionary of summary metrics
    """
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot per-class accuracy
    ax1.bar(range(hp["output_size"]), summary["class_accuracies"])
    ax1.set_xticks(range(hp["output_size"]))
    ax1.set_xlabel("Digit Class")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Per-class Accuracy")

    # Highlight the best and worst classes
    best_class = summary["best_class"]
    worst_class = summary["worst_class"]

    # Change colors for best and worst classes
    bars = ax1.patches
    bars[best_class].set_color("green")
    bars[worst_class].set_color("red")

    # Add text labels
    ax1.text(
        best_class,
        summary["class_accuracies"][best_class] - 1,
        "Best",
        ha="center",
        va="top",
        color="white",
        fontweight="bold",
    )
    ax1.text(
        worst_class,
        summary["class_accuracies"][worst_class] + 1,
        "Worst",
        ha="center",
        va="bottom",
        color="white",
        fontweight="bold",
    )

    # Find y-axis limits that show the differences better
    min_acc = min(summary["class_accuracies"]) - 0.5
    max_acc = 100
    ax1.set_ylim(min_acc, max_acc)

    # Plot most confused pairs
    if summary["confused_pairs"]:
        pairs = summary["confused_pairs"][:5]  # Top 5 pairs
        pair_labels = [f"{p[0]}→{p[1]}" for p in pairs]
        counts = [p[2] for p in pairs]

        ax2.bar(range(len(pairs)), counts)
        ax2.set_xticks(range(len(pairs)))
        ax2.set_xticklabels(pair_labels, rotation=45)
        ax2.set_xlabel("True→Predicted")
        ax2.set_ylabel("Count")
        ax2.set_title("Most Confused Digit Pairs")
    else:
        ax2.text(
            0.5, 0.5, "No confused pairs found", ha="center", va="center"
        )

    plt.tight_layout()
    plt.show()


def show_successful_predictions(model, n_examples=20, test_loader=None):
    """
    Show a grid of correctly classified examples.

    Args:
        model: Trained model
        n_examples: Number of examples to show
        test_loader: Test data loader
    """
    # Load test dataset
    if test_loader is None:
        test_dataset = get_dataset(
            train=False, transform=transforms.ToTensor()
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=100, shuffle=True
        )

    # Set model to evaluation mode
    model.eval()

    # Collect correctly classified examples
    correct_examples = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(hp["device"]), labels.to(hp["device"])
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Find correctly classified examples
            correct_mask = predicted == labels
            correct_indices = correct_mask.nonzero(as_tuple=True)[0]

            # Add to our collection
            for idx in correct_indices:
                if len(correct_examples) < n_examples:
                    correct_examples.append(
                        (
                            images[idx].cpu(),
                            labels[idx].item(),
                            F.softmax(outputs[idx], dim=0).cpu().numpy(),
                        )
                    )
                else:
                    break

            if len(correct_examples) >= n_examples:
                break

    # Create a grid to display the examples
    grid_size = int(np.ceil(np.sqrt(n_examples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    # Plot each example
    for i, (image, label, probs) in enumerate(correct_examples):
        if i < len(axes):
            axes[i].imshow(image.squeeze().numpy(), cmap="gray")
            confidence = probs[label]
            axes[i].set_title(
                f"Digit: {label}\nConf: {confidence:.4f}", color="green"
            )
            axes[i].axis("off")

    # Hide unused subplots
    for i in range(len(correct_examples), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
