import os
import time
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import wandb

from src.config.hyperparameters import hyperparameters as hp
from src.models.network import MNISTClassifier
from src.data.data_loader import get_dataloaders


def find_latest_checkpoint(checkpoint_dir=hp["checkpoints_dir"]):
    """
    Find the latest checkpoint file in the specified directory.

    Args:
        checkpoint_dir: Directory to search for checkpoint files

    Returns:
        Path to the latest checkpoint file, or None if no checkpoint is found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # Get all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

    if not checkpoint_files:
        return None

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    checkpoint = checkpoint_files[0]
    print(f"Found checkpoint: {checkpoint}")

    return checkpoint


def setup_training(model, max_epochs=hp["max_epochs"]):
    """
    Set up PyTorch Lightning training with WandB logging and model checkpointing.

    Args:
        model: PyTorch Lightning model (MNISTClassifier instance)
        max_epochs: Maximum number of training epochs

    Returns:
        PyTorch Lightning Trainer configured with loggers and callbacks
    """
    # # Initialize WandB logger
    wandb_logger = WandbLogger(project=hp["wandb_project"], log_model=True)

    # # Monitor the model
    wandb_logger.watch(model, log="gradients", log_freq=500)
    from pytorch_lightning.loggers import TensorBoardLogger

    # Wait a moment for the run to be properly initialized
    time.sleep(1)

    # Get the run name from WandB so we can save it accordingly
    run_name = wandb.run.name

    print(f"WandB run name: {run_name}")

    # Configure checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=hp["checkpoints_dir"],
        filename=f"{run_name}-{{epoch}}-{{val_acc:.4f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )

    # Configure hardware acceleration
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Training on {accelerator.upper()}")

    # Set up the trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1 if accelerator != "cpu" else None,
        log_every_n_steps=500,
        deterministic=True,
    )

    return trainer


def train_model(hparams=hp):
    """
    Train the MNIST classifier model from scratch.

    Args:
        max_epochs: Maximum number of epochs to train for

    Returns:
        Tuple of (trained_model, best_checkpoint_path)
    """
    print("Starting model training...")

    # Create dataloaders for training and validation
    train_loader, val_loader = get_dataloaders()

    # Initialize the model
    model = MNISTClassifier(
        input_size=hparams["input_size"],
        hidden_size=hparams["hidden_size"],
        output_size=hparams["output_size"],
        learning_rate=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
        dropout_rate=hparams["dropout_rate"],
        conv_channels1=hparams["conv_channels1"],
        conv_channels2=hparams["conv_channels2"],
        conv_channels3=hparams["conv_channels3"],
        kernel_size=hparams["kernel_size"],
        fc_size=hparams["fc_size"],
    )

    # Setup training (WandB, checkpointing, etc.)
    trainer = setup_training(model, max_epochs=hparams["max_epochs"])

    # Start training
    print(f"Training for {hparams['max_epochs']} epochs...")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # After training, find the latest checkpoint
    best_checkpoint_path = find_latest_checkpoint()

    # Finish WandB run
    wandb.finish()

    print(f"Training complete! Best model saved to: {best_checkpoint_path}")
    return model, best_checkpoint_path


def load_model_from_checkpoint(checkpoint_path=None):
    """
    Load a model from a checkpoint or train a new one if no checkpoint exists.

    Args:
        checkpoint_path: Path to checkpoint file, if None will search for one

    Returns:
        The loaded model
    """
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = MNISTClassifier.load_from_checkpoint_custom(
            checkpoint_path, device=device
        )
        return model
    else:
        print("No checkpoint found. Training a new model...")
        model, _ = train_model()
        return model


def list_all_checkpoints(checkpoint_dir=hp["checkpoints_dir"]):
    """
    List all available checkpoints in the specified directory.

    Args:
        checkpoint_dir: Directory to search for checkpoint files

    Returns:
        List of checkpoint file paths sorted by modification time (newest first)
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
        return []

    # Get all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

    if not checkpoint_files:
        print(f"No checkpoint files found in '{checkpoint_dir}'.")
        return []

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    # Print the checkpoints with their metadata
    print(f"Found {len(checkpoint_files)} checkpoint(s):")
    for i, checkpoint in enumerate(checkpoint_files):
        # Extract metadata from the filename
        filename = os.path.basename(checkpoint)

        # Try to parse epoch and accuracy from filename (format: [runname]-[epoch]-[val_acc].ckpt)
        try:
            # Extract information from the filename parts
            parts = filename.replace(".ckpt", "").split("-")
            if len(parts) >= 3:
                # Run name could contain hyphens, so join all but the last two parts
                run_name = "-".join(parts[:-2])
                epoch = parts[-2]
                accuracy = parts[-1]

                # Format the output
                print(f"  {i+1}. {filename}")
                print(
                    f"     Run: {run_name}, Epoch: {epoch}, Accuracy: {accuracy}"
                )
                print(
                    f"     Last Modified: {time.ctime(os.path.getmtime(checkpoint))}"
                )
            else:
                # If filename doesn't match expected format
                print(f"  {i+1}. {filename}")
                print(
                    f"     Last Modified: {time.ctime(os.path.getmtime(checkpoint))}"
                )
        except Exception as e:
            # If there's any error in parsing
            print(f"  {i+1}. {filename}")
            print(
                f"     Last Modified: {time.ctime(os.path.getmtime(checkpoint))}"
            )

    return checkpoint_files


def compare_checkpoints(checkpoint_paths, test_loader=None):
    """
    Compare the performance of multiple checkpoints on the test dataset.

    Args:
        checkpoint_paths: List of checkpoint file paths to compare
        test_loader: DataLoader for the test dataset, created if None

    Returns:
        Dictionary mapping checkpoint paths to their test accuracy
    """
    from src.data.data_loader import load_test_dataset

    if not checkpoint_paths:
        print("No checkpoints to compare.")
        return {}

    # Create test loader if not provided
    if test_loader is None:
        test_loader = load_test_dataset()

    results = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loop through each checkpoint
    for checkpoint_path in checkpoint_paths:
        try:
            # Load the model
            model = MNISTClassifier.load_from_checkpoint_custom(
                checkpoint_path, device=device
            )
            model.eval()

            # Calculate accuracy on test set
            correct = 0
            total = 0

            with torch.inference_mode():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            results[checkpoint_path] = accuracy

            # Print result
            print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
            print(f"Test Accuracy: {accuracy:.4f}%")
            print("-" * 50)

        except Exception as e:
            print(f"Error evaluating checkpoint {checkpoint_path}: {e}")
            results[checkpoint_path] = float("nan")

    # Sort and find the best checkpoint
    if results:
        best_checkpoint = max(results.items(), key=lambda x: x[1])
        print(f"\nBest checkpoint: {os.path.basename(best_checkpoint[0])}")
        print(f"Test Accuracy: {best_checkpoint[1]:.4f}%")

    return results
