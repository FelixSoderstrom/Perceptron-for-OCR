import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

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


def main():
    left_spacer, main_col, right_spacer = st.columns([1, 3, 1])

    with main_col:
        st.title("Neural Network MNIST Digit Classifier (Lightning + WandB)")
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

        if st.session_state.processing:
            st.info("Processing... Please wait")


if __name__ == "__main__":
    main()
