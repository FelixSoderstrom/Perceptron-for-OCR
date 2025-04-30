from pytorch.network import PytorchNeuralNetwork
from data.mnist_loader import choose_number, get_dataloaders
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import glob
import wandb
import time


def pipeline():
    """Pipeline function for running the network using a saved Lightning checkpoint"""
    checkpoint_path = find_latest_checkpoint()

    if not checkpoint_path:
        print("No recent checkpoint found in default WandB log location.")
        print(
            "Attempting to find best model checkpoint saved by ModelCheckpoint callback..."
        )
        checkpoint_path = find_best_model_checkpoint()

    if not checkpoint_path:
        print("No checkpoint found. Training the network first...")
        run_training()
        checkpoint_path = find_latest_checkpoint()
        if not checkpoint_path:
            print("Training finished but still no checkpoint found. Exiting.")
            exit()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        network = PytorchNeuralNetwork.load_from_checkpoint_custom(
            checkpoint_path, device=device_str
        )
        network.eval()
        network.to(device_str)
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        print("Maybe try deleting the checkpoint file and retraining?")
        exit()

    data = choose_number(pytorch=True)
    vector = data["vector"].to(device_str)
    output: list[float] = network.predict(vector)
    print_output(output, data["number"])


def print_output(output: list[float], n: int):
    """Displays the prediction in the terminal"""
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    print("\033c")
    print(
        "Here are the results after passing the "
        f"image through the {RED}PyTorch Lightning{RESET} network"
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


def find_latest_checkpoint(checkpoint_dir="lightning_logs"):
    """Finds the latest .ckpt file in the WandB log directories."""
    try:
        version_dirs = glob.glob(os.path.join(checkpoint_dir, "version_*"))
        if not version_dirs:
            return None

        latest_version_dir = max(version_dirs, key=os.path.getmtime)

        ckpt_files = glob.glob(
            os.path.join(latest_version_dir, "checkpoints", "*.ckpt")
        )
        if not ckpt_files:
            return None

        latest_ckpt = max(ckpt_files, key=os.path.getmtime)
        return latest_ckpt
    except Exception as e:
        print(f"Error finding checkpoint: {e}")
        return None


def find_best_model_checkpoint(
    search_dirs=["pytorch-mnist-ocr", "lightning_logs"],
):
    """Finds the best .ckpt file possibly saved by ModelCheckpoint callback in likely directories."""
    best_ckpt_path = None
    latest_mtime = 0

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        try:
            search_pattern = os.path.join(search_dir, "**", "*.ckpt")
            ckpt_files = glob.glob(search_pattern, recursive=True)

            for ckpt in ckpt_files:
                try:
                    mtime = os.path.getmtime(ckpt)
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        best_ckpt_path = ckpt
                except OSError:
                    continue

        except Exception as e:
            print(f"Error searching for checkpoints in '{search_dir}': {e}")
            continue

    if best_ckpt_path:
        print(f"Found potential best checkpoint: {best_ckpt_path}")
    else:
        print("Could not find any checkpoint file saved by ModelCheckpoint.")

    return best_ckpt_path


def run_training(max_epochs=40):
    """Trains the network using PyTorch Lightning, WandB, and ModelCheckpoint."""
    # Enable tensor core optimization for NVIDIA GPUs (RTX 3090)
    torch.set_float32_matmul_precision("high")

    print(
        f"Starting training for {max_epochs} epochs with PyTorch Lightning, WandB, and ModelCheckpoint..."
    )
    train_loader, val_loader = get_dataloaders()

    network = PytorchNeuralNetwork(learning_rate=1e-3)

    wandb_logger = WandbLogger(project="pytorch-mnist-ocr", log_model=True)
    wandb_logger.watch(network, log="all")

    # Get the WandB run name after the logger is initialized
    # Wait a moment for the run to be properly initialized
    time.sleep(1)

    # Get the run name (like "dazzling-violet-2")
    run_name = wandb.run.name
    print(f"WandB run name: {run_name}")

    # Include the run name in the checkpoint filename
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename=f"{run_name}-{{epoch}}-{{val_acc:.4f}}",
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Training on {accelerator.upper()}")

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1 if accelerator != "cpu" else None,
        log_every_n_steps=50,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model=network,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    wandb.finish()

    print(
        "Training complete! Best checkpoint saved by ModelCheckpoint callback."
    )
