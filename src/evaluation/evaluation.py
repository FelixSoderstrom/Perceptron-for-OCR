import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from src.data.data_loader import load_test_dataset
from src.config.hyperparameters import hyperparameters as hp
from src.models.network import MNISTClassifier
from src.training.trainer import load_model_from_checkpoint


def evaluate_model(model, test_loader=None):
    """
    Evaluate the model on the test dataset.

    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for the test dataset

    Returns:
        Tuple of (accuracy, confusion_matrix, class_accuracies, error_indices)
    """
    # Create test loader if not provided
    if test_loader is None:
        test_loader = load_test_dataset()

    # Ensure model is on the correct device
    model = model.to(hp["device"])

    # Set the model to evaluation mode
    model.eval()

    # Initialize counters and lists
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    error_indices = []  # Store indices of misclassified examples

    # Disable gradient computation for evaluation
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Move tensors to the appropriate device
            images, labels = images.to(hp["device"]), labels.to(hp["device"])

            # Forward pass
            outputs = model(images)

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Update counters
            batch_size = labels.size(0)
            total += batch_size
            correct += (predicted == labels).sum().item()

            # Collect predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Identify misclassified examples
            mistakes = (predicted != labels).nonzero(as_tuple=True)[0]
            for idx in mistakes:
                # Calculate the global index
                global_idx = batch_idx * test_loader.batch_size + idx.item()
                error_indices.append(
                    (global_idx, labels[idx].item(), predicted[idx].item())
                )

    # Calculate overall accuracy
    accuracy = 100 * correct / total

    # Create confusion matrix
    confusion_mat = np.zeros(
        (hp["output_size"], hp["output_size"]), dtype=int
    )
    for pred, label in zip(all_preds, all_labels):
        confusion_mat[label][pred] += 1

    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(hp["output_size"]):
        class_correct = confusion_mat[i][i]
        class_total = np.sum(confusion_mat[i])
        class_accuracies.append(100 * class_correct / class_total)

    # Print evaluation results
    print(f"Test Accuracy: {accuracy:.4f}%")
    print("\nPer-class Accuracy:")
    for i, acc in enumerate(class_accuracies):
        print(f"  Digit {i}: {acc:.4f}%")

    print(f"\nTotal correct classifications: {correct} out of {total}")
    print(f"Total errors: {total - correct}")

    return accuracy, confusion_mat, class_accuracies, error_indices


def analyze_model_confidence(model, test_loader=None, n_samples=500):
    """
    Analyze the model's confidence in its predictions.

    Args:
        model: Trained model
        test_loader: DataLoader for the test dataset
        n_samples: Number of samples to analyze
    """
    # Create test loader if not provided
    if test_loader is None:
        test_loader = load_test_dataset()

    # Ensure model is on the correct device
    model = model.to(hp["device"])

    # Set the model to evaluation mode
    model.eval()

    # Lists to store results
    correct_confidences = []
    incorrect_confidences = []
    top_probs = []

    # Sample counter
    sample_count = 0

    # Disable gradient computation for evaluation
    with torch.inference_mode():
        for images, labels in test_loader:
            # Move tensors to the appropriate device
            images, labels = images.to(hp["device"]), labels.to(hp["device"])

            # Forward pass to get probabilities
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            # Get predicted class and confidence
            max_probs, predicted = torch.max(probs, 1)

            # Collect confidence values for correct and incorrect predictions
            for i in range(len(labels)):
                if sample_count >= n_samples:
                    break

                confidence = max_probs[i].item()
                is_correct = (predicted[i] == labels[i]).item()

                if is_correct:
                    correct_confidences.append(confidence)
                else:
                    incorrect_confidences.append(confidence)

                # Store top probabilities
                top_probs.append(probs[i].cpu().numpy())

                sample_count += 1

            if sample_count >= n_samples:
                break

    # Plot confidence distribution for correct vs incorrect predictions
    plt.figure(figsize=(10, 6))

    bins = np.linspace(0, 1, 21)

    plt.hist(
        correct_confidences,
        bins=bins,
        alpha=0.7,
        label="Correct Predictions",
        color="green",
    )
    plt.hist(
        incorrect_confidences,
        bins=bins,
        alpha=0.7,
        label="Incorrect Predictions",
        color="red",
    )

    plt.xlabel("Confidence (Probability)")
    plt.ylabel("Count")
    plt.title("Distribution of Model Confidence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Analyze average confidence values
    avg_correct_conf = (
        np.mean(correct_confidences) if correct_confidences else 0
    )
    avg_incorrect_conf = (
        np.mean(incorrect_confidences) if incorrect_confidences else 0
    )

    print(
        f"Average confidence for correct predictions: {avg_correct_conf:.4f}"
    )
    print(
        f"Average confidence for incorrect predictions: {avg_incorrect_conf:.4f}"
    )

    # Analyze probability distribution across all classes
    top_probs = np.array(top_probs)
    avg_probs = np.mean(top_probs, axis=0)

    plt.figure(figsize=(10, 5))
    plt.bar(range(hp["output_size"]), avg_probs)
    plt.xlabel("Digit Class")
    plt.ylabel("Average Probability")
    plt.title("Average Probability Distribution Across Classes")
    plt.xticks(range(hp["output_size"]))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return avg_probs, correct_confidences, incorrect_confidences


def summarize_model_performance(model=None, test_loader=None):
    """
    Generate a comprehensive summary of the model's performance.

    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for the test dataset

    Returns:
        Dictionary containing summary metrics
    """
    # Load model if not provided
    if model is None:
        model = load_model_from_checkpoint()
    else:
        # Ensure model is on the correct device
        model = model.to(hp["device"])

    # Load test data if not provided
    if test_loader is None:
        test_loader = load_test_dataset()

    # Evaluate the model
    accuracy, confusion_mat, class_accuracies, error_indices = evaluate_model(
        model, test_loader
    )

    # Calculate additional metrics
    n_errors = len(error_indices)
    error_rate = n_errors / len(test_loader.dataset) * 100

    # Most confused digit pairs
    confused_pairs = []
    for i in range(hp["output_size"]):
        for j in range(hp["output_size"]):
            if i != j and confusion_mat[i][j] > 0:
                confused_pairs.append((i, j, confusion_mat[i][j]))

    # Sort by count in descending order
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    # Summarize model architecture
    if isinstance(model, MNISTClassifier):
        n_conv_layers = len(model.conv_layers)
        n_fc_layers = len(model.fc_layers)
        n_params = sum(p.numel() for p in model.parameters())

        # Calculate approximate inference time
        import time

        start_time = time.time()
        with torch.inference_mode():
            for i in range(10):  # Average over multiple runs
                model(next(iter(test_loader))[0].to(hp["device"]))
        avg_inference_time = (time.time() - start_time) / 10
    else:
        n_conv_layers = "Unknown"
        n_fc_layers = "Unknown"
        n_params = "Unknown"
        avg_inference_time = "Unknown"

    # Compile summary
    summary = {
        "accuracy": accuracy,
        "error_rate": error_rate,
        "class_accuracies": class_accuracies,
        "best_class": np.argmax(class_accuracies),
        "worst_class": np.argmin(class_accuracies),
        "confused_pairs": confused_pairs[:5],  # Top 5 confused pairs
        "n_samples": len(test_loader.dataset),
        "n_errors": n_errors,
        "model_architecture": {
            "conv_layers": n_conv_layers,
            "fc_layers": n_fc_layers,
            "total_parameters": n_params,
            "avg_inference_time": avg_inference_time,
        },
    }

    return summary
