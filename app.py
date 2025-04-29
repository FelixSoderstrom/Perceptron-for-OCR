import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

st.set_page_config(layout="wide")


def get_checkpoint_files(checkpoint_dir="checkpoints"):
    """Return a list of .ckpt checkpoint file paths found in the specified directory."""
    if not os.path.isdir(checkpoint_dir):
        st.warning(
            f"Checkpoint directory '{checkpoint_dir}' not found. Please create it or ensure checkpoints are placed there."
        )
        return []
    try:
        # Search non-recursively for .ckpt files directly in the checkpoint directory
        search_pattern = os.path.join(checkpoint_dir, "*.ckpt")
        ckpt_files = glob.glob(search_pattern)

        # Sort by modification time, newest first (optional, but can be helpful)
        if ckpt_files:
            try:
                ckpt_files.sort(key=os.path.getmtime, reverse=True)
            except Exception:
                pass  # Ignore sorting errors if file times are weird

        return ckpt_files
    except Exception as e:
        st.error(f"Error finding checkpoint files in '{checkpoint_dir}': {e}")
        return []


@st.cache_resource
def initialize_model(checkpoint_path):  # Still takes full path
    """Initialize the model using the selected Lightning checkpoint file from the ./checkpoints dir."""
    try:
        from pytorch.network import (
            PytorchNeuralNetwork,
        )  # Still need the class definition

        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if the provided path exists (it should be a full path already)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            st.error(
                f"Selected checkpoint file not found at: {checkpoint_path}"
            )
            return None, device_str

        # Use the custom loading function from the network class
        network = PytorchNeuralNetwork.load_from_checkpoint_custom(
            checkpoint_path, device=device_str
        )
        network.eval()  # Ensure it's in eval mode
        network.to(device_str)

        st.success(f"Loaded checkpoint: {os.path.basename(checkpoint_path)}")
        return network, device_str  # Return string for device
    except ImportError:
        st.error(
            "Failed to import PytorchNeuralNetwork. Check pytorch/network.py."
        )
        return None, "cpu"
    except Exception as e:
        st.error(
            f"Error initializing model with {os.path.basename(checkpoint_path) if checkpoint_path else 'N/A'}: {str(e)}"
        )
        return None, "cpu"


def plot_digit(image_vector):
    image = image_vector.reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    return plt


def process_digit(number, network, device):
    """Process the selected digit and display predictions"""
    try:
        from data.mnist_loader import get_image

        # Device is now a string ('cpu' or 'cuda')
        device_obj = torch.device(device)

        image_vector = get_image(number, pytorch=True)
        # Ensure image_vector is float and normalized if needed by predict (already done in get_image)
        vector = image_vector.to(device_obj)
        output = network.predict(
            vector
        )  # Predict should handle device internally if needed, or ensure input is on correct device

        col1, col2 = st.columns([1, 2])

        with col1:
            # Ensure numpy conversion happens on CPU
            fig = plot_digit(image_vector.cpu().numpy())
            st.pyplot(fig)

        with col2:
            probabilities = [prob * 100 for prob in output]
            chart_data = {
                "Digit": list(range(10)),
                "Probability (%)": probabilities,
            }

            st.bar_chart(chart_data, x="Digit", y="Probability (%)")

            predicted_digit = np.argmax(probabilities)
            max_probability = max(probabilities)

            if predicted_digit == number:
                st.success(
                    f"✅ Correctly predicted digit {predicted_digit} with {max_probability:.2f}% confidence!"
                )
            else:
                st.error(
                    f"❌ Incorrectly predicted digit {predicted_digit} with {max_probability:.2f}% confidence"
                )
                st.write(f"The actual digit was {number}")
    except Exception as e:
        st.error(f"Error processing digit: {str(e)}")
    finally:
        # Ensure processing flag is reset even if errors occur
        if "processing" in st.session_state:
            st.session_state.processing = False


def load_mnist_test_dataset(batch_size=100):
    """
    Load the MNIST test dataset using torchvision.

    Args:
        batch_size (int): The batch size for the DataLoader

    Returns:
        DataLoader: A DataLoader for the MNIST test dataset
    """
    # Define normalization transform to match our training (-0.5, 0.5)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load the test dataset
    test_dataset = datasets.MNIST(
        root="./data/MNIST", train=False, download=True, transform=transform
    )

    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return test_loader


def run_benchmark(network, device, batch_size=100):
    """
    Run benchmark on all MNIST test images and return accuracy metrics.

    Args:
        network: The neural network model
        device: Device to run the benchmark on ('cpu' or 'cuda')
        batch_size: Batch size for processing test images

    Returns:
        dict: Dictionary containing benchmark results
    """
    try:
        device_obj = torch.device(device)
        network.to(device_obj)
        network.eval()

        # Load test dataset
        test_loader = load_mnist_test_dataset(batch_size)

        total_samples = 0
        correct_predictions = 0
        confusion_matrix = np.zeros((10, 10), dtype=int)
        class_correct = np.zeros(10)
        class_total = np.zeros(10)

        # Create progress bar once
        progress_bar = st.progress(0)
        progress_text = st.empty()
        progress_text.text("Preparing to benchmark...")

        start_time = time.time()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                # Update progress
                progress = (batch_idx + 1) / len(test_loader)
                progress_bar.progress(progress)
                progress_text.text(
                    f"Processing batch {batch_idx+1}/{len(test_loader)} ({progress*100:.1f}%)"
                )

                # Move tensors to the device
                images = images.to(device_obj)
                labels = labels.to(device_obj)

                # Forward pass
                outputs = network(images)

                # Get predictions
                _, predicted = torch.max(outputs, 1)

                # Update counters
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                # Update confusion matrix
                for i in range(labels.size(0)):
                    confusion_matrix[labels[i].item()][
                        predicted[i].item()
                    ] += 1

                # Update per-class accuracy
                for i in range(10):
                    mask = labels == i
                    class_correct[i] += (predicted[mask] == i).sum().item()
                    class_total[i] += mask.sum().item()

        end_time = time.time()
        inference_time = end_time - start_time

        # Clear progress indicators
        progress_text.empty()

        # Calculate metrics
        overall_accuracy = 100 * correct_predictions / total_samples
        per_class_accuracy = np.zeros(10)
        for i in range(10):
            per_class_accuracy[i] = (
                100 * class_correct[i] / class_total[i]
                if class_total[i] > 0
                else 0
            )

        # Return results
        return {
            "overall_accuracy": overall_accuracy,
            "per_class_accuracy": per_class_accuracy,
            "confusion_matrix": confusion_matrix,
            "total_samples": total_samples,
            "inference_time": inference_time,
        }
    except Exception as e:
        st.error(f"Error running benchmark: {str(e)}")
        return None


def display_benchmark_results(results):
    """
    Display benchmark results in a formatted way.

    Args:
        results (dict): Dictionary containing benchmark results
    """
    if not results:
        return

    st.subheader("Benchmark Results")

    # Calculate additional metrics
    per_class_accuracy = results["per_class_accuracy"]
    confusion_matrix = results["confusion_matrix"]

    # Find most correctly and incorrectly guessed digits
    most_correct_digit = int(np.argmax(per_class_accuracy))
    most_correct_accuracy = per_class_accuracy[most_correct_digit]

    # Calculate error rates for each digit (100 - accuracy)
    error_rates = 100 - per_class_accuracy
    most_incorrect_digit = int(np.argmax(error_rates))
    most_incorrect_error = error_rates[most_incorrect_digit]

    # Calculate overall error percentage
    overall_error = 100 - results["overall_accuracy"]

    # Create a layout with metrics on the left, confusion matrix on the right
    metrics_col, matrix_col = st.columns([1, 2])

    with metrics_col:
        # Add some spacing at the top to align better with the confusion matrix title
        st.markdown("<br>", unsafe_allow_html=True)

        # First metric
        st.metric("Overall Accuracy", f"{results['overall_accuracy']:.2f}%")

        # Add spacing between metrics
        st.markdown(
            "<div style='margin-top: 25px;'></div>", unsafe_allow_html=True
        )

        # Second metric
        st.metric("Error Rate", f"{overall_error:.2f}%")

        # Add spacing between metrics
        st.markdown(
            "<div style='margin-top: 25px;'></div>", unsafe_allow_html=True
        )

        # Third metric
        st.metric(
            "Best Recognized Digit",
            f"{most_correct_digit} ({most_correct_accuracy:.2f}%)",
        )

        # Add spacing between metrics
        st.markdown(
            "<div style='margin-top: 25px;'></div>", unsafe_allow_html=True
        )

        # Fourth metric
        st.metric(
            "Most Confused Digit",
            f"{most_incorrect_digit} ({most_incorrect_error:.2f}% error)",
        )

    with matrix_col:
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        # Let confusion matrix use the full width of its column
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        cax = ax.matshow(results["confusion_matrix"], cmap="Blues")
        fig.colorbar(cax)

        # Set labels
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        # Set ticks
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xticklabels(np.arange(10))
        ax.set_yticklabels(np.arange(10))

        # Add text annotations
        for i in range(10):
            for j in range(10):
                text_color = (
                    "white"
                    if results["confusion_matrix"][i, j]
                    > results["confusion_matrix"].max() / 2
                    else "black"
                )
                ax.text(
                    j,
                    i,
                    results["confusion_matrix"][i, j],
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

        st.pyplot(fig)

        # Add explanation of confusion matrix
        st.caption(
            "Confusion Matrix: Rows represent true digits, columns represent predicted digits."
        )


def main():
    left_spacer, main_col, right_spacer = st.columns([1, 3, 1])

    with main_col:
        st.title("Neural Network MNIST Digit Classifier")
        st.write("By Felix Söderström")
        # Get checkpoints from the fixed ./checkpoints/ directory
        available_checkpoints = (
            get_checkpoint_files()
        )  # Gets full paths from ./checkpoints/

        if not available_checkpoints:
            st.error(
                "No Lightning checkpoint (.ckpt) files found in the 'checkpoints' directory."
            )  # Updated error message
            st.error(
                "Please train the model using 'python train.py' first, then copy the desired\n"
                "best checkpoint (e.g., 'best-checkpoint-epoch=X-val_acc=Y.ckpt') from the \n"
                "'pytorch-mnist-ocr/<run_id>/checkpoints/' directory into the 'checkpoints' directory."
            )
            return

        # Create display names (basenames) for the selectbox, map them to full paths
        checkpoint_display_names = {
            os.path.basename(p): p for p in available_checkpoints
        }

        # Default to the first one found (potentially the most recent if sorted)
        default_display_name = list(checkpoint_display_names.keys())[0]

        select_model, confirmation = st.columns([1, 1])
        with select_model:
            selected_display_name = st.selectbox(
                "Select Model Checkpoint:",
                options=list(
                    checkpoint_display_names.keys()
                ),  # Show basenames
                index=0,  # Default to the first one
                key="checkpoint_selector",
            )
            # Get the full path corresponding to the selected display name
            selected_checkpoint_path = checkpoint_display_names.get(
                selected_display_name
            )

        if "processing" not in st.session_state:
            st.session_state.processing = False

        network = None  # Initialize network to None
        device = "cpu"  # Default device

        with confirmation:
            st.markdown("<br>", unsafe_allow_html=True)
            if selected_checkpoint_path:
                # Pass the full path to initialize_model
                network, device = initialize_model(selected_checkpoint_path)

        if network is None:
            if available_checkpoints:
                st.error(
                    "Failed to initialize the selected model. Please check the logs."
                )
            return

        st.write("Select a digit to generate and predict:")

        cols = st.columns(10)

        if cols[0].button(
            "0", disabled=st.session_state.processing, key="btn_0"
        ):
            st.session_state.processing = True
            process_digit(0, network, device)

        if cols[1].button(
            "1", disabled=st.session_state.processing, key="btn_1"
        ):
            st.session_state.processing = True
            process_digit(1, network, device)

        if cols[2].button(
            "2", disabled=st.session_state.processing, key="btn_2"
        ):
            st.session_state.processing = True
            process_digit(2, network, device)

        if cols[3].button(
            "3", disabled=st.session_state.processing, key="btn_3"
        ):
            st.session_state.processing = True
            process_digit(3, network, device)

        if cols[4].button(
            "4", disabled=st.session_state.processing, key="btn_4"
        ):
            st.session_state.processing = True
            process_digit(4, network, device)

        if cols[5].button(
            "5", disabled=st.session_state.processing, key="btn_5"
        ):
            st.session_state.processing = True
            process_digit(5, network, device)

        if cols[6].button(
            "6", disabled=st.session_state.processing, key="btn_6"
        ):
            st.session_state.processing = True
            process_digit(6, network, device)

        if cols[7].button(
            "7", disabled=st.session_state.processing, key="btn_7"
        ):
            st.session_state.processing = True
            process_digit(7, network, device)

        if cols[8].button(
            "8", disabled=st.session_state.processing, key="btn_8"
        ):
            st.session_state.processing = True
            process_digit(8, network, device)

        if cols[9].button(
            "9", disabled=st.session_state.processing, key="btn_9"
        ):
            st.session_state.processing = True
            process_digit(9, network, device)

        st.divider()

        # Add benchmark button
        st.subheader("Benchmark on Test Dataset")
        st.write(
            "Run a benchmark on all 10,000 MNIST test images to evaluate model performance."
        )

        if st.button(
            "Run Benchmark",
            disabled=st.session_state.processing,
            key="btn_benchmark",
        ):
            st.session_state.processing = True
            with st.spinner(
                "Running benchmark on all 10,000 MNIST test images..."
            ):
                benchmark_results = run_benchmark(network, device)

                if benchmark_results:
                    display_benchmark_results(benchmark_results)

            st.session_state.processing = False

        if st.session_state.processing:
            st.info("Processing... Please wait")


if __name__ == "__main__":
    main()
