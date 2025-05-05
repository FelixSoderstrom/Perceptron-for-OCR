import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Tuple, Optional, Any

# Set page title and layout
st.set_page_config(
    page_title="MNIST Model Benchmark Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define constants
CHECKPOINTS_DIR = "../checkpoints"
RESULTS_FILE = "./model_accuracy_scores.json"

# CSS styling
st.markdown(
    """
<style>
    .header-style {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subheader-style {
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #1e3a8a;  /* Dark blue color for better visibility */
    }
    .metric-label {
        font-size: 14px;
        color: #4b5563;  /* Dark gray for better contrast */
        margin-top: 5px;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Title
st.markdown(
    '<div class="header-style">MNIST Model Benchmark Dashboard</div>',
    unsafe_allow_html=True,
)


# Function to load the model from checkpoint
class MNISTClassifier(pl.LightningModule):
    """
    A placeholder class to load the PyTorch Lightning checkpoint.
    We implement enough functionality to load and run the model.
    """

    def __init__(
        self,
        conv_channels1=32,
        conv_channels2=64,
        conv_channels3=128,
        kernel_size=3,
        fc_size=256,
        output_size=10,
        dropout_rate=0.3,
    ):
        super().__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels1,
            kernel_size=kernel_size,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(conv_channels1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=conv_channels1,
            out_channels=conv_channels2,
            kernel_size=kernel_size,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(conv_channels2)

        self.conv3 = nn.Conv2d(
            in_channels=conv_channels2,
            out_channels=conv_channels3,
            kernel_size=kernel_size,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(conv_channels3)

        # Calculate the size of the flattened features
        # Input: 28x28 -> after 3 pools with stride 2: 3x3
        # So the flattened size will be 3*3*conv_channels3
        self.flat_size = 3 * 3 * conv_channels3

        # Save parameters for diagnostics
        self.conv_channels1 = conv_channels1
        self.conv_channels2 = conv_channels2
        self.conv_channels3 = conv_channels3
        self.fc_size = fc_size

        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, fc_size)
        self.fc2 = nn.Linear(fc_size, output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        # Ensure input has the right shape
        if x.dim() == 1:
            # Single flattened image, reshape to [1, 1, 28, 28]
            x = x.view(1, 1, 28, 28)
        elif x.dim() == 2:
            # Batch of flattened images, reshape to [batch_size, 1, 28, 28]
            batch_size = x.size(0)
            x = x.view(batch_size, 1, 28, 28)
        elif x.dim() == 4 and x.size(1) == 1:
            # Already in the correct format [batch_size, 1, 28, 28]
            pass
        else:
            raise ValueError(f"Unexpected input tensor shape: {x.shape}")

        # First convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Third convolutional layer
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # First fully connected layer with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)

        return x

    @staticmethod
    def load_from_checkpoint_custom(checkpoint_path, device="cpu"):
        """
        Load model from checkpoint and move to specified device
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract state dict, handling different formats
        if "state_dict" in checkpoint:
            state_dict = {
                k.replace("model.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            state_dict = checkpoint

        # Determine model architecture from the checkpoint
        # Inspect the state dict to get layer dimensions
        if "conv1.weight" in state_dict:
            conv_channels1 = state_dict["conv1.weight"].shape[
                0
            ]  # Output channels
        else:
            conv_channels1 = 64  # Default if not found

        if "conv2.weight" in state_dict:
            conv_channels2 = state_dict["conv2.weight"].shape[
                0
            ]  # Output channels
        else:
            conv_channels2 = 128  # Default if not found

        if "conv3.weight" in state_dict:
            conv_channels3 = state_dict["conv3.weight"].shape[
                0
            ]  # Output channels
        else:
            conv_channels3 = 256  # Default if not found

        if "fc1.weight" in state_dict:
            fc_size = state_dict["fc1.weight"].shape[0]  # Output neurons
        else:
            fc_size = 128  # Default if not found

        # Get kernel size
        if "conv1.weight" in state_dict:
            kernel_size = state_dict["conv1.weight"].shape[
                2
            ]  # Assuming square kernel
        else:
            kernel_size = 3  # Default if not found

        # Create a model with matching architecture
        model = MNISTClassifier(
            conv_channels1=conv_channels1,
            conv_channels2=conv_channels2,
            conv_channels3=conv_channels3,
            kernel_size=kernel_size,
            fc_size=fc_size,
        )

        # Print model architecture for debugging
        print(f"Created model with architecture:")
        print(f"  conv1: {conv_channels1} channels")
        print(f"  conv2: {conv_channels2} channels")
        print(f"  conv3: {conv_channels3} channels")
        print(f"  fc_size: {fc_size} neurons")

        # Load the state dict
        load_result = model.load_state_dict(state_dict, strict=False)

        # Print any missing or unexpected keys
        if load_result.missing_keys:
            print(f"Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Unexpected keys: {load_result.unexpected_keys}")

        model = model.to(device)
        model.eval()  # Set to evaluation mode

        print(f"Checkpoint loaded from {checkpoint_path} to {device}")
        return model


# Function to get available checkpoint files
def get_checkpoint_files() -> List[str]:
    """Get all checkpoint files from the checkpoints directory"""
    checkpoints_dir = Path(CHECKPOINTS_DIR)

    if not checkpoints_dir.exists():
        return []

    return sorted(
        [
            os.path.join(CHECKPOINTS_DIR, f)
            for f in os.listdir(checkpoints_dir)
            if f.endswith(".ckpt")
        ]
    )


# Function to load the MNIST test dataset
def load_test_dataset(batch_size=100):
    """
    Load the MNIST test dataset using torchvision.

    Args:
        batch_size: The batch size for the DataLoader

    Returns:
        DataLoader: A DataLoader for the MNIST test dataset
    """
    # Define normalization transform to match our training (-0.5, 0.5)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the test dataset
    test_dataset = datasets.MNIST(
        root="../data/MNIST", train=False, download=True, transform=transform
    )

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return test_loader, len(test_dataset)


# Function to evaluate the model
def evaluate_model(model, test_loader):
    """
    Evaluate the model performance on the test dataset.

    Args:
        model: The model to evaluate
        test_loader: DataLoader for the test dataset

    Returns:
        Tuple containing accuracy, confusion matrix, and other metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Initialize variables
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    error_indices = []

    # Process each batch
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Count correct predictions
            batch_size = labels.size(0)
            total += batch_size
            correct += (predicted == labels).sum().item()

            # Collect predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Identify misclassified examples
            mistakes = (predicted != labels).nonzero(as_tuple=True)[0]
            for idx in mistakes:
                error_indices.append(
                    (
                        batch_idx * test_loader.batch_size + idx.item(),
                        labels[idx].item(),
                        predicted[idx].item(),
                    )
                )

    # Calculate metrics
    accuracy = 100 * correct / total
    error_rate = 100 - accuracy
    conf_mat = confusion_matrix(all_labels, all_preds, labels=range(10))

    # Calculate per-class accuracy and identify best/worst performers
    class_correct = np.diag(conf_mat)
    class_total = np.sum(conf_mat, axis=1)
    class_accuracy = 100 * class_correct / class_total

    best_class = np.argmax(class_accuracy)
    worst_class = np.argmin(class_accuracy)

    # Find most confused pairs
    confused_pairs = []
    for i in range(10):
        for j in range(10):
            if i != j and conf_mat[i, j] > 0:
                confused_pairs.append((i, j, conf_mat[i, j]))

    # Sort by count in descending order
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    return {
        "accuracy": accuracy,
        "error_rate": error_rate,
        "confusion_matrix": conf_mat,
        "class_accuracy": class_accuracy,
        "best_class": best_class,
        "worst_class": worst_class,
        "confused_pairs": confused_pairs,
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
    }


# Function to save results to JSON
def save_result_to_json(checkpoint_name, accuracy):
    """Save benchmark results to a JSON file"""
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                results = json.load(f)
        else:
            results = {}

        results[checkpoint_name] = accuracy

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        return True
    except Exception as e:
        st.error(f"Error saving results: {e}")
        return False


# Function to load results from JSON
def load_results_from_json():
    """Load benchmark results from the JSON file"""
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return {}


# Function to plot confusion matrix
def plot_confusion_matrix(conf_mat):
    """
    Plot a confusion matrix as a heatmap.

    Args:
        conf_mat: Confusion matrix from sklearn

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=True)

    # Add labels and ticks
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(10) + 0.5)
    ax.set_yticks(np.arange(10) + 0.5)
    ax.set_xticklabels(range(10))
    ax.set_yticklabels(range(10))

    return fig


# Function to create accuracy per class bar chart
def plot_class_accuracy(class_accuracy):
    """
    Plot accuracy per digit class as a bar chart.

    Args:
        class_accuracy: Array of accuracy values per class

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(10), class_accuracy, color="steelblue")

    # Add value labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.3,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Add labels and title
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by Digit Class")
    ax.set_xticks(range(10))
    ax.set_ylim([min(class_accuracy) - 1, 100.5])  # Set y-axis limits
    ax.grid(axis="y", alpha=0.3)

    return fig


# Create two-column layout
col1, col2 = st.columns([7, 3])

# Main content (left column)
with col1:
    # Checkpoint selection
    checkpoint_files = get_checkpoint_files()

    if not checkpoint_files:
        st.error("No checkpoint files found in the checkpoints directory.")
    else:
        # Get filenames only for display
        checkpoint_names = [os.path.basename(f) for f in checkpoint_files]
        default_idx = 0

        selected_checkpoint_name = st.selectbox(
            "Select a model checkpoint:", checkpoint_names, index=default_idx
        )

        # Get the full path of selected checkpoint
        selected_checkpoint = checkpoint_files[
            checkpoint_names.index(selected_checkpoint_name)
        ]

        # Run benchmark button
        if st.button("Run Benchmark", type="primary"):
            with st.spinner("Running benchmark..."):
                # Load dataset
                test_loader, total_samples = load_test_dataset()
                st.info(f"Test dataset loaded with {total_samples} samples")

                try:
                    # Load model using our custom method
                    model = MNISTClassifier.load_from_checkpoint_custom(
                        selected_checkpoint, device="cpu"
                    )
                    st.success(
                        f"Model loaded from {selected_checkpoint_name}"
                    )

                    # Evaluate model
                    results = evaluate_model(model, test_loader)

                    # Save results
                    save_result_to_json(
                        selected_checkpoint_name, results["accuracy"]
                    )

                    # Display results
                    st.markdown(
                        '<div class="subheader-style">Benchmark Results</div>',
                        unsafe_allow_html=True,
                    )

                    # Show metrics in a row
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-value">{results['accuracy']:.2f}%</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    with metric_cols[1]:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-value">{results['error_rate']:.2f}%</div>
                            <div class="metric-label">Error Rate</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    with metric_cols[2]:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-value">{results['best_class']}</div>
                            <div class="metric-label">Digit {results['best_class']} (Most Accurate)</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    with metric_cols[3]:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-value">{results['worst_class']}</div>
                            <div class="metric-label">Digit {results['worst_class']} (Least Accurate)</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    # Show detailed metrics
                    st.markdown("### Detailed Results")
                    st.write(f"Total samples: {results['total_samples']}")
                    st.write(f"Correct predictions: {results['correct']}")
                    st.write(f"Incorrect predictions: {results['incorrect']}")

                    # Plot confusion matrix
                    st.markdown("### Confusion Matrix")
                    conf_mat_fig = plot_confusion_matrix(
                        results["confusion_matrix"]
                    )
                    st.pyplot(conf_mat_fig)

                    # Plot class accuracy
                    st.markdown("### Accuracy by Digit Class")
                    class_acc_fig = plot_class_accuracy(
                        results["class_accuracy"]
                    )
                    st.pyplot(class_acc_fig)

                except Exception as e:
                    st.error(f"Error during benchmark: {str(e)}")
                    st.exception(e)  # Added to display full traceback

# Leaderboard (right column)
with col2:
    st.markdown(
        '<div class="subheader-style">Leaderboard</div>',
        unsafe_allow_html=True,
    )

    # Load results
    results = load_results_from_json()

    if not results:
        st.info("No benchmark results available. Run a benchmark first!")
    else:
        # Create a DataFrame for display
        data = [
            {"Model": model_name, "Accuracy (%)": f"{accuracy:.4f}"}
            for model_name, accuracy in sorted(
                results.items(), key=lambda x: x[1], reverse=True
            )
        ]

        # Display leaderboard
        st.table(data)
